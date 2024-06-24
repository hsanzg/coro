use smallvec::SmallVec;
use std::alloc::Layout;
use std::arch::{asm, global_asm};
use std::cell::RefCell;
use std::io::Error;
use std::marker::PhantomData;
use std::mem::ManuallyDrop;
use std::ops::{Coroutine, CoroutineState, Deref};
use std::os::raw::c_void;
use std::pin::Pin;
use std::ptr::{addr_of, null_mut, NonNull};
use std::slice::SliceIndex;
use std::sync::OnceLock;
use std::{io, ptr};

macro_rules! bail_if {
    ($e:expr) => {
        if $e {
            return Err(std::io::Error::last_os_error());
        }
    };
}

/// Returns the size of a page in the current architecture.
fn page_size() -> usize {
    static VALUE: OnceLock<usize> = OnceLock::new();
    *VALUE.get_or_init(|| unsafe { libc::sysconf(libc::_SC_PAGESIZE) as usize })
}

/// The direction by which the stack pointer changes after a `push` instruction.
enum StackOrientation {
    /// A program stack grows towards _higher_ memory addresses. In other words,
    /// a `push` instruction _increases_ the stack pointer.
    Upwards,
    /// A program stack grows towards _lower_ memory addresses. In other words,
    /// a `push` instruction _decreases_ the stack pointer.
    Downwards,
}

impl StackOrientation {
    /// The orientation of a program stack in the current platform.
    #[cfg(any(
        target_arch = "x86",
        target_arch = "x86_64",
        target_arch = "arm",
        target_arch = "aarch64",
        target_arch = "riscv32",
        target_arch = "riscv64",
        target_arch = "mips",
        target_arch = "mips64",
        target_arch = "powerpc",
        target_arch = "powerpc64",
        target_arch = "wasm32"
    ))]
    fn current() -> Self {
        Self::Downwards
    }
}

/// The program stack of a coroutine, used to store data and control structures.
/// More precisely, it is a set of contiguous memory pages where a coroutine is
/// free to store any data. There are exceptions to this statement, however:
/// 1. The [`resume`] and [`yield`] operations employ the first eight bytes of
///    a coroutine's stack to record status information during transfers of
///    control. Here the meaning of "first" depends on the [stack growth direction]
///    in the present platform.
/// 2. To protect against overflows, we protect the last page of a program stack.
///    In this way, the process does not have permission to read from, write to,
///    or execute any memory location within this _guard page_; any attempt to
///    do so will cause a protection fault. For this reason, some [authors] do
///    not include the guard page as part of a program stack.
///
/// [`resume`]: `Coro::resume`
/// [`yield`]: `yield_`
/// [stack growth direction]: `StackOrientation`
/// [authors]: https://devblogs.microsoft.com/oldnewthing/20220203-00/?p=106215
struct Stack {
    /// The starting address of the memory region.
    base: NonNull<u8>,
    /// multiple of `page_size()`.
    size: usize,
    // buf: NonNull<c_void>, // todo: change to u8
}

impl Stack {
    /// Allocates space for a new program stack.
    fn new(size: usize) -> Result<Self, Error> {
        // We cannot use the unchecked variant of `from_size_align`: Even if
        // we know that `page_size` is a power of 2, the passed `size` value
        // might overflow an `isize` after alignment.
        let page_size = page_size();
        let size = Layout::from_size_align(size, page_size);
        let ptr = unsafe {
            libc::mmap(
                null_mut(),
                size,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_GROWSDOWN | libc::MAP_STACK,
                -1,
                0,
            )
        };
        bail_if!(ptr.is_null());
        // todo: check that ptr is nonnull (and print errno if not) and change NonNull::new below to new_unchecked.
        // Protect last page. todo: document why we do this.
        let prot_res = unsafe { libc::mprotect(ptr, page_size, libc::PROT_NONE) };
        bail_if!(prot_res == -1);
        Ok(Self {
            base: unsafe { NonNull::new_unchecked(ptr.cast()) },
            size,
        })
    }

    fn base(&self) -> NonNull<u8> {
        self.base
    }

    fn end(&self) -> NonNull<u8> {
        unsafe { self.base.add(self.size) }
    }

    fn guard(&self) -> NonNull<u8> {
        unsafe { self.base.add(page_size()) }
    }
}

impl Drop for Stack {
    fn drop(&mut self) {
        unsafe { libc::munmap(self.base.as_ptr().cast(), self.size) };
    }
}

/// of available stack regions/segments? for new coroutines.
struct StackPool {
    stacks: Vec<Stack>,
}

impl StackPool {
    pub const fn new() -> Self {
        Self { stacks: Vec::new() }
    }

    pub fn take(&mut self, min_size: usize) -> Stack {
        if let Some(ix) = self.stacks.iter().position(|s| s.size >= min_size) {
            self.stacks.swap_remove(ix)
        } else {
            // todo: align to power of 2? Take into account cache aliasing, though
            Stack::new(min_size).expect("failed to create new stack")
        }
    }

    pub fn give(&mut self, stack: Stack) {
        self.stacks.push(stack);
    }
}

thread_local! {
    static STACK_POOL: RefCell<StackPool> = const { RefCell::new(StackPool::new()) };
    // todo: is this the most appropriate type? Can we encode pinning here?
    // todo: get rid of this thread local by keeping parent coroutine information
    //       in the first part of the current coroutine stack. In fact, we only
    //       need some parts of it (which address?)
    static ACTIVE: RefCell<SmallVec<NonNull<Coro>, 4>> = const { RefCell::new(SmallVec::new() )};
}

/// "An independent process that shares and pass data and control
/// back and forth."
pub struct Coro {
    state: CoroutineState<(), ()>,
    stack: ManuallyDrop<Stack>,
    f_ptr: fn(),
}

/*extern "C" resume_trampoline() {

}*/

global_asm!(
    r#"
.global resume_trampoline
resume_trampoline:
    mov rsp, rsi
    push rsi
    call rdi
    pop rsp
"#
);

unsafe extern "C" fn resume_trampoline() {}

impl Coro {
    pub fn new<F>(f: F) -> Self
    where
        F: FnOnce(),
    {
        /*extern "C" fn trampoline<F>()
        where
            F: FnOnce(),
        {
            unsafe { asm!("call rdi") };
            // let f_ptr = unsafe { ptr::read(raw_ptr as *const F) };
            // todo: switch stack pointer to correct place within `f_ptr`.
            // loop {}
        }*/

        let stack = STACK_POOL.with_borrow_mut(|pool| pool.take(0x100));
        let f_addr = addr_of!(f);
        // let f_ptr = (f as *const _) as usize;
        Self {
            state: CoroutineState::Yielded(()),
            stack: ManuallyDrop::new(stack),
            f_ptr: unsafe { std::mem::transmute(f_addr) },
        }
    }
}

impl Drop for Coro {
    fn drop(&mut self) {
        eprintln!("giving back stack to pool");
        STACK_POOL.with_borrow_mut(|pool| {
            // todo: review safety.
            let stack = unsafe { ManuallyDrop::take(&mut self.stack) };
            pool.give(stack);
        });
    }
}

impl Coroutine for Coro {
    type Yield = ();
    type Return = ();

    fn resume(self: Pin<&mut Self>, _arg: ()) -> CoroutineState<Self::Yield, Self::Return> {
        ACTIVE.with_borrow_mut(|active| {
            active.push(NonNull::from(self.deref()));
            eprintln!("pushed coro to active stack");
        });
        //let current_addr = &*self as *const Self as usize;
        let f_ptr = (&*self).f_ptr;
        let stack_ptr = (&*self).stack.end().as_ptr();
        eprintln!("calling coro f_ptr {f_ptr:?} with stack @ {stack_ptr:?}");

        //let f_ptr = self.f_ptr;
        // The stack pointer is guaranteed to be suitably aligned.
        unsafe {
            asm!("call {tramp}", tramp = sym resume_trampoline, in("rdi") f_ptr, in("rsi") stack_ptr, clobber_abi("C"))
        };
        // todo: to see if the coroutine is done, we can check the stack pointer.
        CoroutineState::Complete(())
    }
}

pub fn yield_() {
    ACTIVE.with_borrow_mut(|active| {
        // todo: replace by expect.
        if let Some(active) = active.pop() {
            let active = unsafe { active.as_ref() };
            todo!("yield active coroutine")
        } else {
            // "from the main program"
            panic!("cannot yield while not running within a coroutine");
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn foo() {
        let mut coro = Coro::new(|| {
            eprintln!("started");
            yield_();
            eprintln!("first resume");
            yield_();
            eprintln!("second resume");
        });
        match Pin::new(&mut coro).resume(()) {
            CoroutineState::Yielded(_) => eprintln!("yielded once"),
            CoroutineState::Complete(_) => panic!("completed before"),
        };
        match Pin::new(&mut coro).resume(()) {
            CoroutineState::Yielded(_) => eprintln!("yielded twice"),
            CoroutineState::Complete(_) => panic!("completed before"),
        }
        match Pin::new(&mut coro).resume(()) {
            CoroutineState::Yielded(_) => panic!("yielded thrice"),
            CoroutineState::Complete(_) => eprintln!("completed"),
        }
    }
}

#[cfg_attr(not(windows), path = "itanium.rs")]
mod imp;

pub use self::imp::*;
use crate::stack::StackPtr;
use alloc::boxed::Box;
use core::alloc::AllocError;
use core::any::Any;
use core::fmt::{Display, Formatter};
use core::hint::{assert_unchecked, cold_path, unreachable_unchecked};
use core::marker::PhantomData;
use core::mem::{ManuallyDrop, MaybeUninit};
use core::num::NonZero;
use core::ops::{Deref, DerefMut};
use core::process::abort_immediate;
use core::ptr;
use core::ptr::{DynMetadata, NonNull, Pointee};

/*#[repr(transparent)]
pub struct IndirectBox<T: ?Sized>(Box<Box<T>>);

impl<T: ?Sized> IndirectBox<T> {
  pub fn try_new(inner: Box<T>) -> core::result::Result<Self, AllocError> {
    const {
      assert!(size_of::<Self>() == size_of::<*mut ()>());
    }
    Box::try_new(inner).map(Self)
  }

  pub fn into_box(self) -> Box<T> {
    Box::into_inner(self.0)
  }
}

impl<T: ?Sized> Deref for IndirectBox<T> {
  type Target = T;

  fn deref(&self) -> &Self::Target {
    &self.0
  }
}

impl<T: ?Sized> DerefMut for IndirectBox<T> {
  fn deref_mut(&mut self) -> &mut Self::Target {
    &mut self.0
  }
}

/*struct ThinPtr<P: Pointee + ?Sized>(Box<(NonNull<()>, P::Metadata)>);

impl<P: Pointee + ?Sized> ThinPtr<P> {
  pub fn from_mut(r: &mut P) -> Self {
    let ptr = NonNull::from_mut(r);
    Self(Box::new(ptr.to_raw_parts()))
  }

  pub fn as_non_null(&self) -> NonNull<P> {
    let (data_ptr, meta) = *self.0;
    NonNull::from_raw_parts(data_ptr, meta)
  }
}

impl<P: Pointee + ?Sized> Deref for ThinPtr<P> {
  type Target = P;

  fn deref(&self) -> &Self::Target {
    unsafe { self.as_non_null().as_ref() }
  }
}

impl<P: Pointee + ?Sized> DerefMut for ThinPtr<P> {
  fn deref_mut(&mut self) -> &mut Self::Target {
    unsafe { self.as_non_null().as_mut() }
  }
}*/
/*
fn foo(mut obj: Box<dyn Any + Send>) {
  let ptr = ThinPtr::from_mut(obj.as_mut());
}

#[repr(transparent)]
struct PanicPayloadRef(Box<(NonNull<()>,)>);

impl PanicPayloadRef {
  pub fn new(obj: &(dyn Any + Send)) -> Self {
    todo!()
  }
}

#[repr(transparent)]
struct PanicPayload(NonNull<()>);

impl PanicPayload {
  pub fn new(obj: Box<dyn Any + Send>) -> Self {
    let orig_ptr = Box::as_ptr(&obj);
    let (orig_data, orig_meta) = orig_ptr.to_raw_parts();

    let Ok((layout, data_offset)) = Layout::new::<Metadata>().extend(Layout::for_value(&obj))
    else {
      handle_alloc_error(Layout::new::<()>());
    };
    let meta_ptr = NonNull::new(unsafe { alloc::alloc::alloc(layout) })
      .unwrap_or_else(|| handle_alloc_error(layout))
      .cast::<Metadata>();
    let data_ptr = unsafe { meta_ptr.byte_add(data_offset) }.cast();

    unsafe { meta_ptr.write(orig_meta) };
    unsafe { data_ptr.write(*orig_data) };

    // todo: dealloc `obj`.
    todo!()
  }

  fn metadata_size() -> usize {
    size_of::<Metadata>()
  }

  fn metadata(&self) -> NonNull<Metadata> {
    // todo: subtract `metadata_size`.
    self.0.cast()
  }

  fn data(&self) -> NonNull<()> {
    todo!()
  }

  fn get(&self) -> NonNull<dyn Any + Send> {
    NonNull::from_raw_parts(self.data(), unsafe { self.metadata().read() })
  }
}

impl alloc::panicking::PanicPayload for PanicPayload {
  fn take_box(&mut self) -> Box<dyn Any + Send> {
    todo!()
  }

  fn get(&mut self) -> &(dyn Any + Send) {
    &self.0
  }
}

impl Display for PanicPayload {
  fn fmt(&self, _f: &mut Formatter<'_>) -> core::fmt::Result {
    unreachable!();
  }
}*/

/// # Safety
///
/// `into_ptr` returns an address with the least significant bit unset.
pub unsafe trait PtrRepr {
  fn into_ptr(self) -> *mut ();
  unsafe fn from_ptr(ptr: *mut ()) -> Self;
}

unsafe impl<T> PtrRepr for *mut T {
  fn into_ptr(self) -> *mut () {
    debug_assert!(self.is_aligned_to(2));
    self.cast()
  }

  unsafe fn from_ptr(ptr: *mut ()) -> Self {
    ptr.cast()
  }
}

unsafe impl PtrRepr for () {
  fn into_ptr(self) -> *mut () {
    ptr::null_mut()
  }

  unsafe fn from_ptr(_: *mut ()) -> Self {
    ()
  }
}

unsafe impl<T: ?Sized> PtrRepr for IndirectBox<T> {
  fn into_ptr(self) -> *mut () {
    Box::into_raw(self.0).cast()
  }

  unsafe fn from_ptr(ptr: *mut ()) -> Self {
    Self(unsafe { Box::from_raw(ptr.cast()) })
  }
}

unsafe impl PtrRepr for Exception {
  fn into_ptr(self) -> *mut () {
    match self {
      Exception::ForcedUnwind => ptr::null_mut(),
      Exception::Panic(payload) => payload.into_ptr(),
    }
  }

  unsafe fn from_ptr(ptr: *mut ()) -> Self {
    if ptr.is_null() {
      Exception::ForcedUnwind
    } else {
      Exception::Panic(unsafe { IndirectBox::from_ptr(ptr) })
    }
  }
}*/

/*#[repr(transparent)]
pub struct Moved(MaybeUninit<*mut ()>);

impl Moved {
  const fn is_inline<T>() -> bool {
    size_of::<T>() <= size_of::<*mut ()>()
  }

  pub unsafe fn new<T>(value: &mut ManuallyDrop<T>) -> Self {
    let mut repr = MaybeUninit::<*mut ()>::zeroed();
    if Self::is_inline::<T>() {
      let value = unsafe { ManuallyDrop::take(value) };
      unsafe { repr.as_mut_ptr().cast::<T>().write_unaligned(value) };
    } else {
      repr.write(ptr::from_mut(value).cast());
    }
    Self(repr)
  }

  pub fn into_raw(self) -> usize {
    unsafe { self.0.assume_init() }.expose_provenance()
  }

  pub unsafe fn from_raw(raw: usize) -> Self {
    Self(MaybeUninit::new(ptr::with_exposed_provenance_mut(raw)))
  }

  pub unsafe fn into_inner<T>(self) -> T {
    if Self::is_inline::<T>() {
      unsafe { self.0.as_ptr().cast::<T>().read_unaligned() }
    } else {
      unsafe { self.0.assume_init().cast::<T>().read() }
    }
  }
}*/

pub enum Unwind {
  Forced,
  Panic(Box<dyn Any + Send + 'static>),
}

// todo: try this completion idea again.
/*
#[repr(transparent)]
pub struct Completion(*mut ());

impl Completion {
  const PANIC_TAG: usize = 1;

  pub fn as_ptr(&self) -> *mut () {
    self.0
  }

  pub unsafe fn from_ptr(ptr: *mut ()) -> Self {
    Self(ptr)
  }

  fn panic_payload(&self) -> Option<NonNull<Box<dyn Any + Send + 'static>>> {
    (self.0.addr() & Self::PANIC_TAG != 0)
      .then(|| self.0.map_addr(|a| a & !Self::PANIC_TAG))
      // SAFETY: `a` is nonzero and has its least significant bit unset.
      .map(|p| unsafe { NonNull::new_unchecked(p) }.cast())
  }

  // SAFETY: Only call if not a forced unwind.
  pub unsafe fn ret_addr(self) -> Option<NonNull<()>> {
    let this = ManuallyDrop::new(self);
    if let Some(ptr) = this.panic_payload() {
      cold_path();
      let payload = Box::into_inner(unsafe { Box::from_non_null(ptr) });
      resume(payload);
    }
    NonNull::new(this.0)
  }
}

impl From<CatchResult<StackPtr>> for Completion {
  fn from(res: CatchResult<StackPtr>) -> Self {
    Self(match res {
      Ok(ret_val) => {
        debug_assert_eq!(ret_val.addr().get() & Self::PANIC_TAG, 0);
        ret_val.as_ptr()
      }
      Err(Exception::ForcedUnwind) => ptr::null_mut(),
      Err(Exception::Panic(payload)) => {
        cold_path();
        // todo: ensure alignment; otherwise the assert below can fail.
        let boxed_ptr = Box::into_non_null(Box::new(payload));
        debug_assert_eq!(size_of_val(&boxed_ptr), size_of::<NonNull<()>>());
        debug_assert_eq!(boxed_ptr.addr().get() & Self::PANIC_TAG, 0);
        boxed_ptr.cast().map_addr(|a| a | Self::PANIC_TAG).as_ptr()
      }
    })
  }
}

impl Drop for Completion {
  fn drop(&mut self) {
    if let Some(ptr) = self.panic_payload() {
      drop(unsafe { Box::from_non_null(ptr) });
    }
  }
}*/

/*
#[must_use]
#[repr(transparent)]
pub struct OldRes<R: PtrRepr> {
  ptr: *mut (),
  _phantom: PhantomData<R>,
}

impl<R: PtrRepr> OldRes<R> {
  const ERROR_TAG: usize = 1;

  pub unsafe fn from_raw(ptr: *mut ()) -> Self {
    Self {
      ptr,
      _phantom: PhantomData,
    }
  }

  pub fn into_raw(self) -> *mut () {
    ManuallyDrop::new(self).ptr
  }

  fn ok(r: R) -> Self {
    let ptr = r.into_ptr();
    assert_eq!(ptr.addr() & Self::ERROR_TAG, 0);
    unsafe { Self::from_raw(ptr) }
  }

  fn err(e: Exception) -> Self {
    let ptr = e.into_ptr();
    assert_eq!(ptr.addr() & Self::ERROR_TAG, 0);
    unsafe { Self::from_raw(ptr.map_addr(|a| a | Self::ERROR_TAG)) }
  }

  pub fn is_ok(&self) -> bool {
    (self.ptr.addr() & Self::ERROR_TAG) == 0
  }

  pub fn is_err(&self) -> bool {
    !self.is_ok()
  }

  pub fn is_unwind(&self) -> bool {
    self.ptr.addr() == Self::ERROR_TAG
  }

  pub fn is_panic(&self) -> bool {
    self.is_err() && !self.is_unwind()
  }

  pub unsafe fn ret_val(self) -> R {
    debug_assert!(self.is_ok());
    let ptr = ManuallyDrop::new(self).ptr;
    unsafe { R::from_ptr(ptr) }
  }

  pub unsafe fn exception(self) -> Exception {
    debug_assert!(!self.is_ok());
    let ptr = ManuallyDrop::new(self)
      .ptr
      .map_addr(|a| a & !Self::ERROR_TAG);
    unsafe { Exception::from_ptr(ptr) }
  }
}

impl<R: PtrRepr> Drop for OldRes<R> {
  fn drop(&mut self) {
    if self.is_ok() {
      drop(unsafe { R::from_ptr(self.ptr) });
    } else {
      let ptr = self.ptr.map_addr(|a| a & !Self::ERROR_TAG);
      drop(unsafe { Exception::from_ptr(ptr) });
    }
  }
}*/

type PanicPayload = Box<dyn Any + Send>;

#[must_use]
#[repr(transparent)]
pub struct Completion(*mut ());

impl Completion {
  const UNWIND: usize = 1;

  pub fn into_raw(self) -> usize {
    ManuallyDrop::new(self).0.expose_provenance()
  }

  pub unsafe fn from_raw(raw: usize) -> Self {
    Self(ptr::with_exposed_provenance_mut(raw))
  }

  pub fn is_ok(&self) -> bool {
    self.0.addr() & Self::UNWIND == 0
  }

  #[cold]
  #[inline(never)]
  unsafe fn rethrow(self) {
    let ptr = ManuallyDrop::new(self)
      .0
      .map_addr(|a| a & !Self::UNWIND)
      .cast::<PanicPayload>();
    if !ptr.is_null() {
      cold_path();
      let payload = *unsafe { Box::from_raw(ptr) };
      resume(payload);
    }
  }

  pub fn try_rethrow(self) {
    if !self.is_ok() {
      unsafe { self.rethrow() };
    }
  }

  #[inline]
  pub fn ret_addr(self) -> Option<StackPtr> {
    let ptr = self.0;
    if self.is_ok() {
      let _ = ManuallyDrop::new(self);
      StackPtr::new(ptr)
    } else {
      cold_path();
      unsafe { self.rethrow() };
      unsafe { unreachable_unchecked() };
    }
  }

  /*fn panic_payload(&self) -> Option<NonNull<PanicPayload>> {
    let ptr = self.0.cast::<PanicPayload>();
    (ptr.addr() & Self::UNWIND != 0)
      .then(|| ptr.map_addr(|a| a & !Self::UNWIND))
      .and_then(NonNull::new)
  }

  pub fn ret_addr(self) -> Option<StackPtr> {
    let this = ManuallyDrop::new(self);
    // todo: this is wrong; it returns `None` for forced unwinds.
    if let Some(ptr) = this.panic_payload() {
      cold_path();
      let payload = Box::into_inner(unsafe { Box::from_non_null(ptr) });
      resume(payload);
    }
    NonNull::new(this.0)
  }

  pub fn check_forced_unwind(self) {
    let this = ManuallyDrop::new(self);
    if let Some(ptr) = this.panic_payload() {
      cold_path();
      let payload = Box::into_inner(unsafe { Box::from_non_null(ptr) });
      resume(payload);
    }
  }*/
}

impl From<CatchResult<StackPtr>> for Completion {
  fn from(res: CatchResult<StackPtr>) -> Self {
    Self(match res {
      Ok(ret_addr) => {
        debug_assert_eq!(ret_addr.addr().get() & Self::UNWIND, 0);
        ret_addr.as_ptr()
      }
      Err(Unwind::Forced) => ptr::without_provenance_mut(Self::UNWIND),
      Err(Unwind::Panic(payload)) => {
        cold_path();
        const {
          assert!(align_of::<PanicPayload>() > Self::UNWIND);
        }
        Box::into_raw(Box::new(payload))
          .cast::<()>()
          .map_addr(|a| a | Self::UNWIND)
      }
    })
  }
}

impl Drop for Completion {
  fn drop(&mut self) {
    let ptr = self.0;
    if !self.is_ok() {
      let ptr = ptr.map_addr(|a| a & !Self::UNWIND).cast::<PanicPayload>();
      if !ptr.is_null() {
        drop(unsafe { Box::from_raw(ptr) });
      }
    }
  }
}

pub type CatchResult<R> = Result<R, Unwind>;

pub fn catch<F: FnOnce() -> R, R>(f: F) -> CatchResult<R> {
  union Data<F, T> {
    f: ManuallyDrop<F>,
    ok: ManuallyDrop<T>,
    err: ManuallyDrop<Unwind>,
  }

  let mut data = Data {
    f: ManuallyDrop::new(f),
  };

  #[inline]
  fn do_call<F: FnOnce() -> R, R>(data: *mut Data<F, R>) {
    let data = unsafe { &mut *data };
    let f = unsafe { ManuallyDrop::take(&mut data.f) };
    data.ok = ManuallyDrop::new(f());
  }

  /*#[cold]
  unsafe fn cleanup(payload: *mut u8) -> IndirectBox<dyn Any + Send + 'static> {
    let obj = unsafe { __rust_panic_cleanup(payload) };
    IndirectBox::try_new(obj)
      .ok()
      .unwrap_or_else(|| abort("failed to allocate boxed ptr to panic payload"))
  }*/

  #[inline]
  #[rustc_nounwind]
  fn do_catch<F: FnOnce() -> T, T>(data: *mut Data<F, T>, payload: *mut u8) {
    let err = if unsafe { is_unwind_exception(payload) } {
      Unwind::Forced
    } else {
      Unwind::Panic(unsafe { __rust_panic_cleanup(payload) })
    };
    unsafe { (*data).err = ManuallyDrop::new(err) };
  }

  if unsafe { !core::intrinsics::catch_unwind(do_call::<F, R>, &raw mut data, do_catch::<F, R>) } {
    Ok(ManuallyDrop::into_inner(unsafe { data.ok }))
  } else {
    Err(ManuallyDrop::into_inner(unsafe { data.err }))
  }
}

// Coroutine returns a NonNull<()>.
// In the case of a panic, we also need to return *const u8 for payload.
// This last payload pointer is aligned. Tag the most significant bit to
// indicate which variant of pointer?

// In `force_unwind`, we normally return `()` but on panic we also
// need a *const u8 for the payload.

/*pub trait PtrRepr {
  // the returned pointer has its least significant bit unset.
  fn into_ptr(self) -> *mut u8;
  unsafe fn from_ptr(ptr: *mut u8) -> Self;
}

impl<T> PtrRepr for *mut T {
  fn into_ptr(self) -> *mut u8 {
    /*const {
      assert!(align_of::<T>() > 1);
    }*/
    self.cast()
  }

  unsafe fn from_ptr(ptr: *mut u8) -> Self {
    ptr.cast()
  }
}

impl PtrRepr for () {
  fn into_ptr(self) -> *mut u8 {
    ptr::null_mut()
  }

  unsafe fn from_ptr(_: *mut u8) -> Self {
    ()
  }
}

pub enum Exception {
  Panic(Box<dyn Any + Send + 'static>),
  ForcedUnwind,
}

impl PtrRepr for Exception {
  fn into_ptr(self) -> *mut u8 {
    match self {
      Exception::Panic(payload) => {
        cold_path();
        Box::into_raw(payload).cast()
      }
      Exception::ForcedUnwind => ptr::null_mut(),
    }
  }

  unsafe fn from_ptr(ptr: *mut u8) -> Self {
    if let Some(ptr) = NonNull::new(ptr) {
      cold_path();
      Exception::Panic(unsafe { Box::from_non_null(ptr) })
    } else {
      Exception::ForcedUnwind
    }
  }
}

// todo: convert to union `ok`, `err`?
// todo: remove impl block and rename to `Outcome`?
#[repr(transparent)]
pub struct TryCatch<R> {
  result: *mut u8,
  _phantom: PhantomData<R>,
}

impl<R: PtrRepr> TryCatch<R> {
  const IS_ERR: usize = 1;
  const EXCEPTION_MASK: usize = !Self::IS_ERR;

  pub fn new<F: FnOnce() -> R>(f: F) -> Self {
    union Data<F, R> {
      f: ManuallyDrop<F>,
      ok: ManuallyDrop<R>,
      err: ManuallyDrop<Exception>,
    }

    let mut data = Data {
      f: ManuallyDrop::new(f),
    };

    #[inline]
    fn do_call<F: FnOnce() -> R, R>(data: *mut Data<F, R>) {
      let data = unsafe { &mut *data };
      let f = unsafe { ManuallyDrop::take(&mut data.f) };
      data.ok = ManuallyDrop::new(f());
    }

    #[inline]
    #[rustc_nounwind]
    fn do_catch<F: FnOnce() -> R, R>(data: *mut Data<F, R>, payload: *mut u8) {
      let err = if unsafe { is_unwind_exception(payload) } {
        Exception::ForcedUnwind
      } else {
        cold_path();
        let obj = unsafe { __rust_panic_cleanup(payload) };
        Exception::Panic(obj)
      };
      unsafe { (*data).err = ManuallyDrop::new(err) };
    }

    let ok =
      unsafe { !core::intrinsics::catch_unwind(do_call::<F, R>, &raw mut data, do_catch::<F, R>) };
    Self {
      result: if ok {
        ManuallyDrop::into_inner(unsafe { data.ok }).into_ptr()
      } else {
        ManuallyDrop::into_inner(unsafe { data.err })
          .into_ptr()
          .map_addr(|p| p | Self::IS_ERR)
      },
      _phantom: PhantomData,
    }
  }

  pub unsafe fn data(&self) -> *mut u8 {
    self.result
  }

  pub unsafe fn exception(&self) -> Exception {
    debug_assert!(!self.is_ok());
    let payload = self.result.map_addr(|p| p & Self::EXCEPTION_MASK);
    unsafe { Exception::from_ptr(payload) }
  }

  pub fn is_ok(&self) -> bool {
    (self.result.addr() & Self::IS_ERR) == 0
  }

  #[cold]
  #[inline(never)]
  pub unsafe fn resume_catch(&self) -> ! {
    match unsafe { self.exception() } {
      Exception::Panic(payload) => resume(payload),
      Exception::ForcedUnwind => unsafe { unreachable_unchecked() },
    }
  }
}*/

/*pub enum Exception {
  Panic(Box<dyn Any + Send + 'static>),
  ForcedUnwind,
}

pub fn catch<T, F: FnOnce() -> T>(f: F) -> Result<T, Exception> {
  union Data<F, T> {
    f: ManuallyDrop<F>,
    ok: ManuallyDrop<T>,
    err: ManuallyDrop<Exception>,
  }

  let mut data = Data {
    f: ManuallyDrop::new(f),
  };

  #[inline]
  fn do_call<F: FnOnce() -> T, T>(data: *mut Data<F, T>) {
    let data = unsafe { &mut *data };
    let f = unsafe { ManuallyDrop::take(&mut data.f) };
    data.ok = ManuallyDrop::new(f());
  }

  #[inline]
  #[rustc_nounwind]
  fn do_catch<F: FnOnce() -> T, T>(data: *mut Data<F, T>, payload: *mut u8) {
    let err = if unsafe { is_unwind_exception(payload) } {
      Exception::ForcedUnwind
    } else {
      cold_path();
      Exception::Panic(unsafe { __rust_panic_cleanup(payload) })
    };
    unsafe { (*data).err = ManuallyDrop::new(err) };
  }

  if unsafe { !core::intrinsics::catch_unwind(do_call::<F, T>, &raw mut data, do_catch::<F, T>) } {
    Ok(ManuallyDrop::into_inner(unsafe { data.ok }))
  } else {
    Err(ManuallyDrop::into_inner(unsafe { data.err }))
  }
}*/

unsafe extern "Rust" {
  #[rustc_std_internal_symbol]
  fn __rust_panic_cleanup(payload: *mut u8) -> Box<dyn Any + Send + 'static>;

  #[rustc_std_internal_symbol]
  safe fn __rust_start_panic(payload: &mut dyn alloc::panicking::PanicPayload) -> u32;
}

pub fn resume(payload: Box<dyn Any + Send>) -> ! {
  struct RewrapBox(Box<dyn Any + Send>);

  impl alloc::panicking::PanicPayload for RewrapBox {
    fn take_box(&mut self) -> Box<dyn Any + Send> {
      core::mem::replace(&mut self.0, Box::new(()))
    }

    fn get(&mut self) -> &(dyn Any + Send) {
      &*self.0
    }
  }

  impl Display for RewrapBox {
    fn fmt(&self, _f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
      unreachable!()
    }
  }

  let _code = __rust_start_panic(&mut RewrapBox(payload));
  // todo: write `code`.
  abort("failed to initiate panic");
}

#[cold]
pub fn abort(_message: &str) -> ! {
  abort_immediate();
}

use crate::stack::StackPtr;
use crate::{Bottom, Control};
use core::arch::naked_asm;
use core::ptr::NonNull;

pub unsafe extern "C-unwind" fn transfer_control(target: &mut Control) {}

#[unsafe(naked)]
pub extern "C-unwind" fn trampoline<R, F>(_control: NonNull<Control>, _ret_addr: NonNull<()>) -> !
where
  F: FnOnce() -> R + Unpin,
{
  naked_asm!("")
}

#[unsafe(naked)]
pub unsafe extern "C-unwind" fn return_control<R>(
  _control: &mut Control,
  _ret_val_addr: NonNull<R>,
) -> ! {
  naked_asm!("")
}

pub fn initial_sp<F>(bottom: NonNull<Bottom<F>>) -> StackPtr {
  todo!()
}

pub unsafe fn force_unwind(control: &mut Control) {
  todo!()
}

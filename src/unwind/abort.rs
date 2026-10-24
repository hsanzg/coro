use core::convert::Infallible;

pub type Exception = *mut Infallible;

pub unsafe fn is_coro_unwind(_: Exception) -> bool {
  unreachable!()
}

pub unsafe fn rethrow(_: Exception) -> ! {
  unreachable!()
}

pub unsafe fn destroy(_: Exception) {
  unreachable!()
}

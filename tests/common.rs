use coro::Coro;

pub fn new_coro<R, F: FnOnce() -> R + Unpin>(f: F) -> Coro<R, F> {
  cfg_select! {
    feature = "common_pool" => Coro::new(f),
    feature = "alloc" => {
      use coro::stack::Stack;
      let stack = Stack::allocate(coro::DEFAULT_STACK_SIZE);
      Coro::with_stack(f, stack)
    }
  }
}

pub fn assert_unstarted<R, F: FnOnce() -> R + Unpin>(coro: &Coro<R, F>) {
  assert!(!coro.has_started());
  assert!(coro.is_paused());
  assert!(!coro.has_finished());
  assert!(!coro.has_panicked());
}

pub fn assert_paused<R, F: FnOnce() -> R + Unpin>(coro: &Coro<R, F>) {
  assert!(coro.has_started());
  assert!(coro.is_paused());
  assert!(!coro.has_finished());
  assert!(!coro.has_panicked());
}

pub fn assert_complete<R, F: FnOnce() -> R + Unpin>(coro: &Coro<R, F>, panicked: bool) {
  assert!(coro.has_started());
  assert!(!coro.is_paused());
  assert_eq!(coro.has_finished(), !panicked);
  assert_eq!(coro.has_panicked(), panicked);
}

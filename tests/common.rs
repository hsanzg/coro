use coro::Coro;

pub fn new_coro<R, F: FnOnce() -> R + Unpin>(f: F) -> Coro<R> {
  // todo: handle non-alloc case.
  cfg_select! {
    feature = "common_pool" => Coro::new(f),
    feature = "stack_alloc" => {
      use coro::stack::Stack;
      let stack = Stack::allocate(coro::DEFAULT_STACK_SIZE);
      Coro::with_stack(f, stack)
    }
    _ => panic!("this test requires the `stack_alloc` feature"),
  }
}

#[allow(unused)]
pub fn assert_unstarted<R>(coro: &Coro<R>) {
  assert!(!coro.has_started());
  assert!(coro.is_resumable());
  assert!(!coro.has_finished());
  assert!(!coro.has_panicked());
}

#[allow(unused)]
pub fn assert_paused<R>(coro: &Coro<R>) {
  assert!(coro.has_started());
  assert!(coro.is_resumable());
  assert!(!coro.has_finished());
  assert!(!coro.has_panicked());
}

#[allow(unused)]
pub fn assert_complete<R>(coro: &Coro<R>, panicked: bool) {
  assert!(coro.has_started());
  assert!(!coro.is_resumable());
  assert_eq!(coro.has_finished(), !panicked);
  assert_eq!(coro.has_panicked(), panicked);
}

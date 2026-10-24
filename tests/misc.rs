mod common;

use common::*;
use core::cell::Cell;
use core::mem::DropGuard;

#[test]
fn unstarted_drops_closure_on_drop() {
  let dropped = Cell::new(false);
  let guard = DropGuard::new((), |_| dropped.set(true));
  drop(new_coro(move || drop(guard)));
  assert!(dropped.get());
}

#[test]
#[cfg(all(
  feature = "stack_alloc",
  not(feature = "common_pool"),
  panic = "unwind"
))]
fn reuse_coro_stacks() {
  use coro::{Coro, suspend};

  let coro = new_coro(|| {});
  let stack = coro.drop_and_take_stack(); // unstarted.
  let (base, size) = (stack.base(), stack.size());

  let mut coro = Coro::with_stack(|| suspend(), stack);
  coro.resume();

  let stack = coro.drop_and_take_stack(); // unfinished.
  assert_eq!(stack.base(), base);
  assert_eq!(stack.size(), size);

  let mut coro = Coro::with_stack(|| {}, stack);
  coro.resume();

  let stack = coro.drop_and_take_stack(); // finished.
  assert_eq!(stack.base(), base);
  assert_eq!(stack.size(), size);
}

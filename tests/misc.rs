mod common;

use common::new_coro;
use std::cell::Cell;
use std::mem::DropGuard;

#[test]
fn unstarted_drops_closure_on_drop() {
  let dropped = Cell::new(false);
  let guard = DropGuard::new((), |_| dropped.set(true));
  drop(new_coro(move || drop(guard)));
  assert!(dropped.get());
}

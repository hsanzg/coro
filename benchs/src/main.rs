use coro::{Coro, suspend};
use corosensei::{Coroutine, Yielder};
use criterion::{Criterion, criterion_group, criterion_main};

fn switch(c: &mut Criterion) {
  let mut coro = Coro::new(|| {
    loop {
      suspend();
    }
  });
  c.bench_function("coro", |b| b.iter(|| coro.resume()));

  let mut coro = Coroutine::new(|yielder, ()| {
    loop {
      yielder.suspend(());
    }
  });
  c.bench_function("corosensei", |b| b.iter(|| coro.resume(())));
}

// todo: repeat this benchmark, but without stack pooling.
fn call(c: &mut Criterion) {
  c.bench_function("coro", |b| {
    b.iter(|| {
      Coro::new(|| {}).resume();
    })
  });

  c.bench_function("corosensei", |b| {
    b.iter(|| {
      Coroutine::new(|_yielder: &Yielder<(), ()>, ()| {}).resume(());
    })
  });
}

criterion_group!(benchs, switch /*call*/,);
criterion_main!(benchs);

// A live game: the host page gives a function that returns what was played since a cursor (a promise of
// { chunks: [{ f: frames as base64, d: decisions, title }], next, restart, closed }); the new frames are
// added to the recording as they come.

NV.live = {
  cursor: 0,
  ended: false,
  // Resolves once the first frames have arrived.
  start(poll) {
    const rec = NV.rec;
    let first;
    const ready = new Promise((resolve) => (first = resolve));
    const ask = async () => {
      if (NV.stopped) return first(); // closed while waiting for the first frames: the start ends too
      if (this.ended) return;
      try {
        const answer = await poll(this.cursor);
        for (const chunk of answer.chunks || []) {
          rec.append(new Int16Array(NV.data.bytes(chunk.f).buffer), chunk.d || []);
          if (chunk.title) {
            rec.title = chunk.title;
            NV.emit("title", { title: chunk.title });
          }
        }
        this.cursor = answer.next;
        if (rec.count) first();
        if (answer.closed) {
          this.ended = true;
          rec.live = false;
          NV.player.atLive = false;
          NV.emit("live-ended");
          return;
        }
      } catch {
        // The host was busy or went away for a moment: ask again.
      }
      setTimeout(ask, 50);
    };
    ask();
    return ready;
  },
};

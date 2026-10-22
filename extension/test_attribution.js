/*
 * Checks for the per-participant attribution in content.js.
 *
 *   node extension/test_attribution.js
 *
 * This is the part that must not be wrong. A score comes from ONE mixed audio
 * stream, so crediting it to a named person is only defensible while Meet says
 * exactly one person has the floor. If that rule leaks -- crediting during
 * overlap, or when the speaking signal is unreadable -- the panel starts
 * printing confident numbers against the wrong face, which is worse than
 * printing nothing.
 *
 * content.js is a content script, not a module, so it is run inside a vm with
 * the browser globals stubbed and its top-level functions read back off the
 * sandbox.
 */

const fs = require("fs");
const path = require("path");
const vm = require("vm");

/* ---- the fake Meet page ------------------------------------------------ */

function tile({ id, name, speaking, speakingReadable = true }) {
  const sig = speakingReadable
    ? { getAttribute: (a) => (a === "data-is-speaking" ? String(!!speaking) : null),
        offsetParent: speaking ? {} : null,
        getClientRects: () => (speaking ? [{}] : []) }
    : null;
  return {
    getAttribute: (a) => (a === "data-participant-id" ? id
                        : a === "data-tooltip" ? name : null),
    querySelector: (sel) => {
      if (sel === "img") return null;
      if (sel === "[data-self-name]") {
        return { getAttribute: () => name, textContent: name };
      }
      if (sel === ".IisKdb" || sel.includes("speaking")) return sig;
      return null;
    },
    textContent: name,
  };
}

function makeSandbox(tiles) {
  const noop = () => {};
  const fakeEl = () => ({
    className: "", id: "", textContent: "", tabIndex: 0, title: "", hidden: false,
    style: { setProperty: noop }, classList: { add: noop, toggle: noop, contains: () => false },
    append: noop, appendChild: noop, addEventListener: noop,
    querySelector: () => null, querySelectorAll: () => [],
    getAttribute: () => null, setAttribute: noop,
  });

  const sandbox = {
    console,
    setInterval, clearInterval, setTimeout, clearTimeout,
    performance: { now: () => Date.now() },
    requestAnimationFrame: () => 0,
    cancelAnimationFrame: noop,
    JSON, Math, Number, String, Object, Array, Date, Promise,
    window: { addEventListener: noop, devicePixelRatio: 1 },
    document: {
      body: { appendChild: noop, contains: () => false },
      createElement: fakeEl,
      querySelector: () => null,
      querySelectorAll: (sel) =>
        sel === "[data-participant-id]" ? tiles : [],
    },
    chrome: {
      runtime: { onMessage: { addListener: noop }, sendMessage: noop },
      storage: { local: { get: () => Promise.resolve({}), set: noop } },
    },
  };
  sandbox.globalThis = sandbox;
  return sandbox;
}

function load(tiles) {
  const src = fs.readFileSync(
    path.join(__dirname, "content.js"), "utf8");
  const sandbox = makeSandbox(tiles);
  vm.createContext(sandbox);
  vm.runInContext(src, sandbox);
  // `let`/`const` at script top level land in the context's global LEXICAL
  // scope, not on the global object, so sandbox.participants is undefined.
  // Evaluating in the same context reaches them.
  sandbox.__eval = (expr) => vm.runInContext(expr, sandbox);
  sandbox.__attr = (v) => vm.runInContext(`attribute(${v})`, sandbox);
  return sandbox;
}

/* ---- checks ------------------------------------------------------------ */

let failures = 0;
function check(name, fn) {
  try {
    fn();
    console.log("ok  " + name);
  } catch (e) {
    failures++;
    console.log("FAIL " + name + "\n     " + e.message);
  }
}
function assert(cond, msg) {
  if (!cond) throw new Error(msg || "assertion failed");
}

check("reads participants and their names", () => {
  const s = load([
    tile({ id: "a", name: "Suryansh Thapliyal", speaking: false }),
    tile({ id: "b", name: "Anubhava Jangwan", speaking: false }),
  ]);
  s.__eval('readParticipants()');
  assert(s.__eval('participants.size') === 2, `found ${s.__eval('participants.size')}, want 2`);
  assert(s.__eval("participants.get('a').name") === "Suryansh Thapliyal",
         "name not read: " + s.__eval("participants.get('a').name"));
});

check("credits the sole speaker", () => {
  const s = load([
    tile({ id: "a", name: "Solo", speaking: true }),
    tile({ id: "b", name: "Quiet", speaking: false }),
  ]);
  s.__eval('readParticipants()');
  s.__attr(0.9);
  assert(s.__eval("participants.get('a').last") === 0.9, "speaker was not credited");
  assert(s.__eval("participants.get('b').last") === null, "silent person was credited");
});

check("credits NOBODY when two people talk over each other", () => {
  const s = load([
    tile({ id: "a", name: "One", speaking: true }),
    tile({ id: "b", name: "Two", speaking: true }),
  ]);
  s.__eval('readParticipants()');
  s.__attr(0.95);
  for (const p of s.__eval('[...participants.values()]')) {
    assert(p.last === null, `${p.name} was credited from overlapping speech`);
  }
});

check("credits NOBODY when nobody is speaking", () => {
  const s = load([tile({ id: "a", name: "Silent", speaking: false })]);
  s.__eval('readParticipants()');
  s.__attr(0.8);
  assert(s.__eval("participants.get('a').last") === null, "credited during silence");
});

check("credits NOBODY when Meet's speaking signal is unreadable", () => {
  const s = load([
    tile({ id: "a", name: "Unknown", speaking: true, speakingReadable: false }),
  ]);
  s.__eval('readParticipants()');
  assert(s.__eval('speakerReadable') === false, "claimed the signal was readable");
  s.__attr(0.99);
  assert(s.__eval("participants.get('a').last") === null,
         "credited a score with no usable speaker signal");
});

check("drops participants who leave", () => {
  const s = load([tile({ id: "a", name: "Stays", speaking: false }),
                  tile({ id: "b", name: "Leaves", speaking: false })]);
  s.__eval('readParticipants()');
  assert(s.__eval('participants.size') === 2);
  s.document.querySelectorAll = (sel) =>
    sel === "[data-participant-id]" ? [tile({ id: "a", name: "Stays", speaking: false })] : [];
  s.__eval('readParticipants()');
  assert(s.__eval('participants.size') === 1, "left participant still listed");
  assert(s.__eval("participants.has('a')"));
});

check("reports an unreadable roster instead of inventing one", () => {
  const s = load([]);
  s.__eval('readParticipants()');
  assert(s.__eval('domReadable') === false, "claimed the roster was readable");
  assert(s.__eval('participants.size') === 0);
});

check("per-person history stays bounded", () => {
  const s = load([tile({ id: "a", name: "Talker", speaking: true })]);
  s.__eval('readParticipants()');
  for (let i = 0; i < 200; i++) s.__attr(0.5);
  assert(s.__eval("participants.get('a').scores.length") === 40,
         `history grew to ${s.__eval("participants.get('a').scores.length")}`);
});


/* ---- redraw guard ------------------------------------------------------ */
/* The graph must not repaint when nothing about the picture has changed.
   Every repaint is a full clear plus two shadow-blurred passes, and doing
   that on every state push is what reads as flicker. */

function fakeCanvas(calls) {
  const ctx = new Proxy({}, {
    get: (_t, k) => {
      if (k === "createLinearGradient") return () => ({ addColorStop() {} });
      if (k === "setTransform") return () => { };
      if (k === "clearRect") return () => { calls.clears++; };
      if (k === "measureText") return () => ({ width: 10 });
      return () => { };
    },
    set: () => true,
  });
  return { clientWidth: 300, clientHeight: 146, width: 0, height: 0,
           getContext: () => ctx };
}

check("graph skips the repaint when the data has not changed", () => {
  const s = load([tile({ id: "a", name: "A", speaking: false })]);
  const calls = { clears: 0 };
  s.__fake = fakeCanvas(calls);
  s.__eval("canvas = __fake");

  const series = "[0.1,0.2,0.3,0.4,0.5]";
  s.__eval(`drawChart(${series}, 0.5)`);
  const afterFirst = calls.clears;
  assert(afterFirst > 0, "first draw did not paint at all");

  s.__eval(`drawChart(${series}, 0.5)`);
  s.__eval(`drawChart(${series}, 0.5)`);
  assert(calls.clears === afterFirst,
         `repainted ${calls.clears - afterFirst} time(s) with unchanged data`);
});

check("graph DOES repaint when a new window arrives", () => {
  const s = load([tile({ id: "a", name: "A", speaking: false })]);
  const calls = { clears: 0 };
  s.__fake = fakeCanvas(calls);
  s.__eval("canvas = __fake");

  s.__eval("drawChart([0.1,0.2,0.3], 0.3)");
  const before = calls.clears;
  s.__eval("drawChart([0.1,0.2,0.3,0.9], 0.9)");
  assert(calls.clears > before, "new data did not trigger a repaint");
});

check("graph repaints when a threshold slider moves", () => {
  const s = load([tile({ id: "a", name: "A", speaking: false })]);
  const calls = { clears: 0 };
  s.__fake = fakeCanvas(calls);
  s.__eval("canvas = __fake");

  s.__eval("drawChart([0.1,0.2,0.3], 0.3)");
  const before = calls.clears;
  s.__eval("RED_AT = 0.8; drawChart([0.1,0.2,0.3], 0.3)");
  assert(calls.clears > before, "moving a threshold did not repaint the bands");
});

console.log(failures ? `\n${failures} check(s) FAILED` : "\nall checks passed");
process.exit(failures ? 1 : 0);

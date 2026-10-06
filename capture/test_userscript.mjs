// Stub-browser tests for capture/tweet-index.user.js.
//
// Run with:  node capture/test_userscript.mjs
//
// There is no browser here: window, document, localStorage, Blob, history and
// fetch are all stubbed, and the real fixture pages are fed in as if X had
// returned them.
//
// The stub models the thing that actually broke pagination: a timeline page is
// many screens tall, and X only requests the next page when you are near the
// bottom of what is already rendered. Scrolling by one screen at a time never
// gets there. Three page layouts are modelled:
//
//   'window'    - the page itself scrolls (what X normally does)
//   'container' - the page cannot scroll; a div holds the timeline
//   'frozen'    - nothing scrolls at all
//
// What this still cannot prove: that the fetch patch beats X's first request
// in Firefox, or that X's real DOM heights behave like this model.

import { readFileSync, readdirSync } from 'fs';
import assert from 'assert';

const SRC = readFileSync('capture/tweet-index.user.js', 'utf8');

/** Paths X actually serves, plus the spelling the design document used. */
const PATHS = {
  bookmarks: '/i/history',
  likes: '/i/history/likes',          // what X really serves
};

/**
 * Synthetic response pages, in the shape X nests them.
 *
 * The original suite read real captures from the author's account. These
 * tests only ever count pages and tweet ids -- nothing reads the content of
 * a tweet -- so generated payloads test exactly the same behaviour without
 * shipping anyone's saved posts. Built in JS so this file still runs with
 * nothing but node.
 */
function makePage(feed, pageNumber, tweetsPerPage = 20) {
  const base = (feed === 'bookmarks' ? 1000 : 2000) + pageNumber * 10;
  const entries = [];
  for (let i = 0; i < tweetsPerPage; i++) {
    const id = `${base}${String(i).padStart(15, '0')}`;
    entries.push({
      entryId: `tweet-${id}`,
      sortIndex: `19000000000000000${99 - pageNumber * 10 - i}`,
      content: {
        entryType: 'TimelineTimelineItem',
        itemContent: {
          itemType: 'TimelineTweet',
          tweet_results: {
            result: {
              __typename: 'Tweet',
              rest_id: id,
              legacy: { full_text: `Post ${id}`, entities: { urls: [] } },
              core: { user_results: { result: {
                core: { screen_name: 'someone', name: 'Someone' } } } },
            },
          },
        },
      },
    });
  }
  entries.push({
    entryId: `cursor-bottom-${pageNumber}`,
    content: { entryType: 'TimelineTimelineCursor',
               value: `cursor-${pageNumber}`, cursorType: 'Bottom' },
  });
  // The two feeds are nested differently, and the userscript classifies a
  // response by its shape rather than by the GraphQL operation name. Getting
  // this wrong means a page is silently ignored, so the generated payloads
  // use the real shapes: bookmarks under data.bookmark_timeline_v2, likes
  // under data.user.result.timeline with scribeConfig.page 'favorites'.
  const instructions = [{ type: 'TimelineAddEntries', entries }];
  if (feed === 'bookmarks') {
    return { data: { bookmark_timeline_v2: { timeline: { instructions } } } };
  }
  return { data: { user: { result: { timeline: { timeline: {
    instructions,
    metadata: { scribeConfig: { page: 'favorites' } },
  } } } } } };
}

function pagesFor(feed, count = 3) {
  return Array.from({ length: count }, (_, n) => makePage(feed, n));
}
const FIXTURES = { bookmarks: pagesFor('bookmarks'), likes: pagesFor('likes') };

/** Timeline entry ids anywhere in a payload. */
function tweetIdsIn(node, out = []) {
  if (Array.isArray(node)) { node.forEach(n => tweetIdsIn(n, out)); return out; }
  if (!node || typeof node !== 'object') return out;
  if (typeof node.entryId === 'string' && /^tweet-\d+$/.test(node.entryId)) {
    out.push(node.entryId);
    return out;
  }
  for (const v of Object.values(node)) tweetIdsIn(v, out);
  return out;
}

/**
 * Pages in a download that carry tweets. A run that reaches the end of a
 * timeline also saves the empty response that told it so, verbatim, which is
 * right but should not count as a page of content.
 */
function contentPages(blob) {
  return blob.pages.filter(p => tweetIdsIn(p.json).length > 0).length;
}

/** The same rule the userscript uses, for the stub's own bookkeeping. */
function kindOf(path) {
  const p = path.replace(/\/+$/, '') || '/';
  if (p.startsWith('/i/history/like')) return 'likes';
  if (p === '/i/history') return 'bookmarks';
  return null;
}

const CLIENT = 900;        // viewport height
const PAGE_PX = 8000;      // rendered height of one page of 20 tweets
const NEAR_BOTTOM = 1200;  // X asks for more within this distance of the end

/**
 * A fresh stubbed browser. `store` persists across runs, like localStorage.
 * `opts.path` overrides the starting URL; `opts.mode` picks the page layout.
 */
function browser(feed, store, opts = {}) {
  // endsCleanly: once the fixtures run out, answer one more request with a
  // page holding no tweets - a timeline saying "nothing further down". X does
  // send this; confirmed on a real bookmarks run, October 2026. Set it false
  // for a timeline that simply goes quiet, which is indistinguishable from a
  // stalled load and must NOT be treated as the end.
  const { path, mode = 'window', endsCleanly = true } =
    typeof opts === 'string' ? { path: opts } : opts;
  const state = { downloads: [], confirmAnswer: true, logs: [], scrolls: 0,
                  clockSkew: 0, nextPage: { bookmarks: 0, likes: 0 },
                  sentEmpty: { bookmarks: false, likes: false } };

  // --- the rendered page ---------------------------------------------------
  const view = {
    top: 0,
    loaded: 0,                                  // pages rendered so far
    get height() { return Math.max(CLIENT, this.loaded * PAGE_PX); },
  };

  function setTop(v) {
    state.scrolls++;
    if (mode === 'frozen' || mode === 'wedged') return;   // nothing moves
    view.top = Math.max(0, Math.min(v, view.height - CLIENT));
    if (view.top + CLIENT >= view.height - NEAR_BOTTOM) deliver();
  }

  // `extra` may carry getters and setters, so it is applied with property
  // descriptors. Spreading it would evaluate each getter once and copy the
  // resulting value, freezing scrollHeight and dropping the scrollTop setter.
  const el = (tag, extra = null) => {
    const node = {
    tagName: tag, style: {}, children: [], className: '', textContent: '',
    title: '', disabled: false, onclick: null, _parent: null,
    scrollHeight: 0, clientHeight: 0, scrollTop: 0, _overflowY: 'visible',
    append(...c) { for (const x of c) this.appendChild(x); },
    appendChild(c) { c._parent = this; this.children.push(c); return c; },
    remove() {
      if (!this._parent) return;
      const i = this._parent.children.indexOf(this);
      if (i >= 0) this._parent.children.splice(i, 1);
      this._parent = null;
    },
    click() { if (!this.disabled && this.onclick) this.onclick(); },
    querySelector(sel) {
      const want = sel.replace('.', '');
      return this.children.find(c => c.className === want) || { textContent: '' };
    },
    };
    if (extra)
      Object.defineProperties(node, Object.getOwnPropertyDescriptors(extra));
    return node;
  };

  const body = el('body');

  // The page element. In 'window' mode it is the thing that scrolls.
  const documentElement = el('html', {
    get scrollHeight() {
      // 'wedged' is a tall page whose scroll position never changes.
      return (mode === 'window' || mode === 'wedged') ? view.height : CLIENT;
    },
    get clientHeight() { return CLIENT; },
    get scrollTop() { return mode === 'window' ? view.top : 0; },
    set scrollTop(v) { if (mode === 'window' || mode === 'wedged') setTop(v); },
  });

  // In 'container' mode the timeline lives in its own scrolling div.
  const container = el('div', {
    get scrollHeight() { return view.height; },
    get clientHeight() { return CLIENT; },
    get scrollTop() { return view.top; },
    set scrollTop(v) { setTop(v); },
    _overflowY: 'auto',
  });

  const realSetTimeout = setTimeout;
  const listeners = {};
  const RealDate = Date;
  class TestDate extends RealDate {          // lets a test reach the time cap
    static now() { return RealDate.now() + state.clockSkew; }
  }

  const sandbox = {
    localStorage: {
      getItem: k => (store.has(k) ? store.get(k) : null),
      setItem: (k, v) => store.set(k, v),
      removeItem: k => store.delete(k),
    },
    location: { pathname: path || PATHS[feed] },
    history: { pushState() {}, replaceState() {} },
    Blob: class { constructor(parts) { this.text = parts.join(''); } },
    URL: { createObjectURL: b => (state.downloads.push(b.text), 'blob:stub') },
    document: {
      body, documentElement, scrollingElement: documentElement,
      readyState: 'loading', createElement: t => el(t),
      querySelectorAll: () => (mode === 'container' ? [container] : []),
      addEventListener: (ev, fn) => {
        if (ev === 'DOMContentLoaded') sandbox.document._ready = fn;
      },
    },
    getComputedStyle: e => ({ overflowY: e._overflowY || 'visible' }),
    addEventListener: (ev, fn) => { listeners[ev] = fn; },
    console: { log: (...a) => state.logs.push(a.join(' ')) },
    innerHeight: CLIENT,
    scrollTo: (x, y) => setTop(y),
    scrollBy: (x, y) => setTop(view.top + y),
    setTimeout: fn => realSetTimeout(fn, 0),          // skip the 2-3s pacing
    setInterval: () => 0,                             // no polling in tests
    confirm: () => state.confirmAnswer,
    Date: TestDate,
    fetch: async (url, json) => ({ clone: () => ({ json: async () => json }) }),
    XMLHttpRequest: class { open() {} addEventListener() {} },
  };
  sandbox.window = sandbox;

  /** Deliver the next page of whichever timeline is currently on screen. */
  function deliver() {
    const k = kindOf(sandbox.location.pathname);
    if (!k) return;
    const list = FIXTURES[k];
    if (state.nextPage[k] >= list.length) {
      if (endsCleanly && !state.sentEmpty[k]) {
        state.sentEmpty[k] = true;
        sandbox.window.fetch('https://x.com/i/api/graphql/abc/Timeline',
                             emptyPayload(k));
      }
      return;
    }
    view.loaded++;                                    // the DOM grows
    sandbox.window.fetch('https://x.com/i/api/graphql/abc/Timeline',
                         list[state.nextPage[k]++]);
  }

  /** A well-formed response for this feed that contains only a cursor. */
  function emptyPayload(feed) {   // hoisted: deliver() calls this
    const entries = [{ entryId: 'cursor-bottom-0', content: {} }];
    const instructions = [{ type: 'TimelineAddEntries', entries }];
    return feed === 'bookmarks'
      ? { data: { bookmark_timeline_v2: { timeline: { instructions } } } }
      : { data: { user: { result: { timeline: { timeline: {
            metadata: { scribeConfig: { page: 'favorites' } },
            instructions } } } } } };
  }

  // Run the userscript with these globals in scope.
  const names = Object.keys(sandbox);
  new Function(...names, SRC)(...names.map(n => sandbox[n]));
  sandbox.document._ready();

  const findPanel = () => body.children.find(c => c.children.length >= 4);

  const api = {
    state, store, deliver, view,
    deliverEmpty: () => sandbox.window.fetch(
      'https://x.com/i/api/graphql/abc/Timeline',
      emptyPayload(kindOf(sandbox.location.pathname))),
    skipAhead: ms => { state.clockSkew += ms; },
    logs: () => state.logs,
    /**
     * Wait for any run in progress to finish.
     *
     * The click handlers do not hand back their promise, so this polls
     * instead: a run holds the buttons disabled, and setTimeout is stubbed to
     * fire immediately, so a whole capture completes in a few ticks. Waiting
     * on the condition rather than a fixed delay keeps the suite quick.
     */
    settle: async (maxTicks = 5000) => {
      const tick = () => new Promise(r => realSetTimeout(r, 0));
      for (let i = 0; i < maxTicks; i++) {
        await tick();
        const p = findPanel();
        if (!p || !p.children[0].disabled) break;
      }
      await tick();                  // let trailing callbacks land
      await tick();
    },
    knownIds: f => JSON.parse(store.get('ti.known.' + (f || feed)) || '[]'),
    totalPages: f => FIXTURES[f || feed].length,
    pagesLoaded: () => view.loaded,
    /** Switch tabs the way X does: change the URL, no reload. */
    navigate(to) {
      sandbox.location.pathname = to;
      sandbox.history.pushState({}, '', to);     // patched by the script
    },
    popstate(to) {
      sandbox.location.pathname = to;
      if (listeners.popstate) listeners.popstate();
    },
    lastDownload: () => JSON.parse(state.downloads[state.downloads.length - 1]),
    status: () => {
      const p = findPanel();
      return p ? p.children[3].textContent : null;
    },
  };
  Object.defineProperties(api, {
    panel: { get: findPanel },
    full:  { get: () => (findPanel() || { children: [] }).children[0] },
    test:  { get: () => (findPanel() || { children: [] }).children[1] },
    reset: { get: () => (findPanel() || { children: [] }).children[2] },
  });
  return api;
}

const results = [];
async function check(name, fn) {
  try { await fn(); results.push(['PASS', name]); }
  catch (e) { results.push(['FAIL', name, e.message]); }
}

// ---------------------------------------------------------------------------
// pagination: the reason a real test run only ever captured one page
// ---------------------------------------------------------------------------

await check('a full run reaches every page of a tall timeline', async () => {
  const b = browser('bookmarks', new Map());
  b.deliver();                                   // the page's first request
  await b.settle();
  assert.strictEqual(b.pagesLoaded(), 1, 'setup: one page rendered');

  b.full.click();
  await b.settle();
  const pages = contentPages(b.lastDownload());
  assert.strictEqual(pages, b.totalPages(),
    `captured ${pages} of ${b.totalPages()} pages - pagination stalled`);
});

await check('one page is many screens tall, so one screen is not enough',
            async () => {
  // Guards the model itself: if a page were shorter than a screen, the test
  // above would pass even with one-screen scrolling and prove nothing.
  const b = browser('bookmarks', new Map());
  b.deliver();
  await b.settle();
  assert.ok(PAGE_PX / CLIENT > 5,
            `a page is only ${(PAGE_PX / CLIENT).toFixed(1)} screens tall`);
});

await check('each step lands at the bottom of what is rendered', async () => {
  const b = browser('bookmarks', new Map());
  b.deliver();
  await b.settle();
  b.test.click();
  await b.settle();
  // Two pages rendered means the first jump reached the bottom of page 1.
  assert.ok(b.pagesLoaded() >= 2,
            `only ${b.pagesLoaded()} page(s) rendered`);
});

await check('a run ends at the bottom, not on a false idle', async () => {
  const b = browser('bookmarks', new Map());
  b.deliver();
  await b.settle();
  b.full.click();
  await b.settle();
  assert.strictEqual(b.view.top, b.view.height - CLIENT,
                     'the run did not finish at the bottom of the page');
});

await check('scrolling a container when the page itself cannot scroll',
            async () => {
  const b = browser('bookmarks', new Map(), { mode: 'container' });
  b.deliver();
  await b.settle();
  b.full.click();
  await b.settle();
  const pages = contentPages(b.lastDownload());
  assert.strictEqual(pages, b.totalPages(),
    `captured ${pages} of ${b.totalPages()} pages via the container`);
});

await check('a page that cannot scroll at all stops instead of looping',
            async () => {
  const b = browser('bookmarks', new Map(), { mode: 'frozen' });
  b.deliver();
  await b.settle();
  b.full.click();
  await b.settle();
  assert.strictEqual(b.state.downloads.length, 1, 'it should still download');
  assert.strictEqual(b.lastDownload().pages.length, 1, 'just the one page');
});

await check('the status line shows scroll position and page height', async () => {
  const b = browser('bookmarks', new Map());
  b.deliver();
  await b.settle();
  b.full.onclick();                              // start without awaiting
  await new Promise(r => setTimeout(r, 30));
  assert.match(b.status(), /scroll \d+\/\d+/,
               `status was: ${b.status()}`);
  await b.settle();
});

// ---------------------------------------------------------------------------
// why a run stopped: one test per reason
// ---------------------------------------------------------------------------

await check('stop reason: the page limit of a test run', async () => {
  const b = browser('bookmarks', new Map());
  b.deliver();
  await b.settle();
  b.test.click();
  await b.settle();
  assert.match(b.status(), /Stopped: the 2-page test limit\./,
               `status was: ${b.status()}`);
});

await check('stop reason: 40 already-saved tweets in a row', async () => {
  const store = new Map();
  const first = browser('bookmarks', store);      // learn the ids
  first.deliver();
  await first.settle();
  first.full.click();
  await first.settle();

  const second = browser('bookmarks', store);     // the weekly delta run
  second.deliver();
  await second.settle();
  second.full.click();
  await second.settle();
  assert.match(second.status(), /Stopped: 40 already-saved tweets in a row/,
               `status was: ${second.status()}`);
  assert.match(second.status(), /everything newer has been captured/);
});

await check('stop reason: the end of the timeline', async () => {
  const b = browser('bookmarks', new Map());
  b.deliver();
  await b.settle();
  b.full.onclick();                               // start, then run dry
  await new Promise(r => setTimeout(r, 20));
  b.deliverEmpty();                               // X: nothing further down
  await b.settle();
  assert.match(b.status(), /Stopped: the end of the timeline\./,
               `status was: ${b.status()}`);
});

await check('stop reason: idle at the bottom with nothing new', async () => {
  // The fixtures run out and X never answers with an empty page, so the run
  // ends on the idle counter rather than on a definite end-of-timeline.
  const b = browser('bookmarks', new Map(), { endsCleanly: false });
  b.deliver();
  await b.settle();
  b.full.click();
  await b.settle();
  assert.match(b.status(), /Stopped: 6 scrolls at the bottom with nothing new/,
               `status was: ${b.status()}`);
});

await check('stop reason: the page stopped scrolling', async () => {
  const b = browser('bookmarks', new Map(), { mode: 'wedged' });
  b.deliver();
  await b.settle();
  b.full.click();
  await b.settle();
  assert.match(b.status(), /Stopped: the page stopped scrolling/,
               `status was: ${b.status()}`);
});

await check('stop reason: the time cap', async () => {
  const b = browser('bookmarks', new Map());
  b.deliver();
  await b.settle();
  b.full.onclick();
  b.skipAhead(21 * 60 * 1000);                    // past the 20-minute cap
  await b.settle();
  assert.match(b.status(), /Stopped: the 20-minute time cap\./,
               `status was: ${b.status()}`);
});

await check('stop reason: the tab changed', async () => {
  const b = browser('bookmarks', new Map());
  b.deliver();
  await b.settle();
  b.full.onclick();
  b.navigate('/i/history/likes');
  await b.settle();
  assert.match(b.status(), /Stopped: the tab changed to another page\./,
               `status was: ${b.status()}`);
});

await check('every stopping condition still saves a file', async () => {
  // Whatever ended the run, the pages gathered so far are worth keeping: a
  // partial capture ingests fine and the next run picks up the rest.
  const cases = {
    'page limit': async b => { b.test.click(); await b.settle(); },
    'idle at bottom': async b => { b.full.click(); await b.settle(); },
    'end of timeline': async b => {
      b.full.onclick();
      await new Promise(r => setTimeout(r, 20));
      b.deliverEmpty();
      await b.settle();
    },
    'time cap': async b => {
      b.full.onclick();
      b.skipAhead(21 * 60 * 1000);
      await b.settle();
    },
    'tab changed': async b => {
      b.full.onclick();
      b.navigate('/i/history/likes');
      await b.settle();
    },
  };
  for (const [name, run] of Object.entries(cases)) {
    const b = browser('bookmarks', new Map());
    b.deliver();
    await b.settle();
    await run(b);
    assert.strictEqual(b.state.downloads.length, 1, `${name}: no file saved`);
    assert.ok(b.lastDownload().pages.length > 0, `${name}: empty file`);
    assert.match(b.status(), /Check Downloads\./, `${name}: ${b.status()}`);
  }
});

await check('a wedged page still saves what it managed to get', async () => {
  const b = browser('bookmarks', new Map(), { mode: 'wedged' });
  b.deliver();
  await b.settle();
  b.full.click();
  await b.settle();
  assert.strictEqual(b.state.downloads.length, 1);
  assert.strictEqual(b.lastDownload().pages.length, 1);
  assert.match(b.status(), /Check Downloads\./);
});

// The known-ID list is the weekly delta's starting point, and a run always
// begins at the newest tweet. So a run that stopped in the middle must not
// record what it saw: next time those IDs would form a wall of 40 knowns in
// the first page or two, the run would stop there, and everything below where
// this run gave up would never be captured. Same trap as the test-run bug.
await check('a time-capped run leaves the known list unchanged', async () => {
  const store = new Map();
  const b = browser('bookmarks', store);
  b.deliver();
  await b.settle();
  b.full.onclick();
  b.skipAhead(21 * 60 * 1000);                    // cut off mid-timeline
  await b.settle();

  assert.match(b.status(), /Stopped: the 20-minute time cap\./);
  assert.match(b.status(),
               /Known ids NOT changed \(run did not reach saved tweets\)/,
               `status was: ${b.status()}`);
  assert.strictEqual(store.has('ti.known.bookmarks'), false,
                     'a cut-short run must not record known ids');
  assert.strictEqual(b.state.downloads.length, 1, 'the file is still saved');
});

await check('a full run after a time-capped one captures every page', async () => {
  const store = new Map();

  const cut = browser('bookmarks', store);        // run 1: cut short
  cut.deliver();
  await cut.settle();
  cut.full.onclick();
  cut.skipAhead(21 * 60 * 1000);
  await cut.settle();
  assert.ok(contentPages(cut.lastDownload()) < cut.totalPages(),
            'setup: run 1 should not have finished the timeline');

  const full = browser('bookmarks', store);       // run 2: unobstructed
  full.deliver();
  await full.settle();
  full.full.click();
  await full.settle();

  const pages = contentPages(full.lastDownload());
  assert.strictEqual(pages, full.totalPages(),
    `run 2 captured ${pages} of ${full.totalPages()} pages - run 1's known `
    + 'ids blocked it');
});

await check('reaching the end of the timeline DOES record known ids', async () => {
  const store = new Map();
  const b = browser('bookmarks', store);          // endsCleanly: true
  b.deliver();
  await b.settle();
  b.full.click();
  await b.settle();
  assert.match(b.status(), /Stopped: the end of the timeline\./);
  assert.doesNotMatch(b.status(), /Known ids NOT changed/);
  assert.ok(b.knownIds().length > 0, 'a completed run should remember ids');
});

await check('meeting 40 known tweets DOES record known ids', async () => {
  const store = new Map();
  const first = browser('bookmarks', store);
  first.deliver();
  await first.settle();
  first.full.click();
  await first.settle();

  const second = browser('bookmarks', store);     // the weekly delta run
  second.deliver();
  await second.settle();
  second.full.click();
  await second.settle();
  assert.match(second.status(), /Stopped: 40 already-saved tweets in a row/);
  assert.doesNotMatch(second.status(), /Known ids NOT changed/);
  assert.ok(second.knownIds().length > 0);
});

await check('idle, stuck and tab-change runs all leave the list alone',
            async () => {
  const cases = {
    'idle at bottom': [{ endsCleanly: false }, async b => {
      b.full.click();
      await b.settle();
    }],
    'stuck': [{ mode: 'wedged' }, async b => {
      b.full.click();
      await b.settle();
    }],
    'tab changed': [{}, async b => {
      b.full.onclick();
      b.navigate('/i/history/likes');
      await b.settle();
    }],
  };
  for (const [name, [opts, run]] of Object.entries(cases)) {
    const store = new Map();
    const b = browser('bookmarks', store, opts);
    b.deliver();
    await b.settle();
    await run(b);
    assert.strictEqual(store.has('ti.known.bookmarks'), false,
                       `${name} recorded known ids: ${b.status()}`);
    assert.match(b.status(), /Known ids NOT changed/, `${name}: ${b.status()}`);
    assert.strictEqual(b.state.downloads.length, 1,
                       `${name}: the file should still be saved`);
  }
});

await check('every run reports some reason', async () => {
  for (const mode of ['window', 'container', 'frozen', 'wedged']) {
    const b = browser('bookmarks', new Map(), { mode });
    b.deliver();
    await b.settle();
    b.full.click();
    await b.settle();
    assert.match(b.status(), /Stopped: /, `mode ${mode}: ${b.status()}`);
    assert.doesNotMatch(b.status(), /unrecognised/, `mode ${mode}`);
  }
});

// ---------------------------------------------------------------------------
// panel basics
// ---------------------------------------------------------------------------

await check('panel has four elements: full, test, reset, status', async () => {
  const b = browser('bookmarks', new Map());
  assert.strictEqual(b.full.textContent, 'Capture bookmarks');
  assert.strictEqual(b.test.textContent, 'Test (2 pages)');
  assert.strictEqual(b.reset.textContent, 'Reset known');
});

// ---------------------------------------------------------------------------
// capture behaviour
// ---------------------------------------------------------------------------

await check('test run downloads a file and stops at 2 pages', async () => {
  const b = browser('bookmarks', new Map());
  b.deliver();
  await b.settle();
  b.test.click();
  await b.settle();
  assert.strictEqual(b.state.downloads.length, 1, 'expected one download');
  const blob = b.lastDownload();
  assert.strictEqual(blob.kind, 'bookmarks');
  assert.strictEqual(blob.pages.length, 2, `got ${blob.pages.length} pages`);
});

await check('test run records NO known ids', async () => {
  const b = browser('bookmarks', new Map());
  b.deliver();
  await b.settle();
  b.test.click();
  await b.settle();
  assert.strictEqual(b.store.has('ti.known.bookmarks'), false,
                     'test run must not write ti.known.bookmarks');
});

await check('full run DOES record known ids', async () => {
  const b = browser('bookmarks', new Map());
  b.deliver();
  await b.settle();
  b.full.click();
  await b.settle();
  assert.ok(b.knownIds().length > 0, 'full run should remember ids');
});

// The bug this guards against: a test run that saved known ids would give the
// next full run a known streak of 40 at the top of the timeline, and
// stopAfterKnown is 40, so the full run would quit after two pages.
await check('full run after a test run is NOT cut short', async () => {
  const store = new Map();

  const t = browser('bookmarks', store);
  t.deliver();
  await t.settle();
  t.test.click();
  await t.settle();
  assert.strictEqual(t.lastDownload().pages.length, 2);

  const f = browser('bookmarks', store);
  f.deliver();
  await f.settle();
  f.full.click();
  await f.settle();
  const fullPages = contentPages(f.lastDownload());
  assert.strictEqual(fullPages, f.totalPages(),
    `full run stopped after ${fullPages} of ${f.totalPages()} pages - `
    + 'the test run poisoned the known-id list');
});

await check('a full run DOES stop early on a genuinely known timeline', async () => {
  const store = new Map();

  const first = browser('bookmarks', store);
  first.deliver();
  await first.settle();
  first.full.click();
  await first.settle();
  assert.ok(first.knownIds().length >= 40,
            `only ${first.knownIds().length} ids remembered`);

  const second = browser('bookmarks', store);
  second.deliver();
  await second.settle();
  second.full.click();
  await second.settle();
  const pages = contentPages(second.lastDownload());
  assert.ok(pages < second.totalPages(),
    `expected an early stop, captured all ${pages} pages`);
});

await check('a second run does not re-download the first run\'s pages', async () => {
  const b = browser('bookmarks', new Map());
  b.deliver();
  await b.settle();

  b.test.click();
  await b.settle();
  const first = b.lastDownload();
  assert.strictEqual(first.pages.length, 2);

  b.test.click();
  await b.settle();
  const second = b.lastDownload();

  assert.strictEqual(b.state.downloads.length, 2);
  const firstStamps = first.pages.map(p => p.capturedAt);
  const reused = second.pages.filter(p => firstStamps.includes(p.capturedAt));
  assert.strictEqual(reused.length, 0,
    `run 2 re-included ${reused.length} page(s) from run 1`);
});

// ---------------------------------------------------------------------------
// switching tabs without a reload
// ---------------------------------------------------------------------------

await check('panel relabels when X switches to the likes tab', async () => {
  const b = browser('bookmarks', new Map());
  assert.strictEqual(b.full.textContent, 'Capture bookmarks');
  b.navigate('/i/history/likes');
  assert.ok(b.panel, 'panel disappeared after navigating');
  assert.strictEqual(b.full.textContent, 'Capture likes');
});

await check('panel relabels on back/forward navigation', async () => {
  const b = browser('likes', new Map());
  assert.strictEqual(b.full.textContent, 'Capture likes');
  b.popstate('/i/history');
  assert.strictEqual(b.full.textContent, 'Capture bookmarks');
});

await check('panel is removed when navigating off the timelines', async () => {
  const b = browser('bookmarks', new Map());
  assert.ok(b.panel);
  b.navigate('/home');
  assert.strictEqual(b.panel, undefined, 'panel should be gone on /home');
});

await check('panel comes back when navigating onto a timeline', async () => {
  const b = browser('bookmarks', new Map(), { path: '/home' });
  assert.strictEqual(b.panel, undefined);
  b.navigate('/i/history/likes');
  assert.ok(b.panel, 'panel should appear on the likes tab');
  assert.strictEqual(b.full.textContent, 'Capture likes');
});

// This is the failure that produced two identical capture_bookmarks_ files:
// the panel was built on the bookmarks page, the user switched to likes
// without reloading, and the run still captured bookmarks.
await check('after switching to likes, a run captures LIKES', async () => {
  const b = browser('bookmarks', new Map());
  b.deliver();
  await b.settle();

  b.navigate('/i/history/likes');
  b.deliver();
  await b.settle();

  b.test.click();
  await b.settle();

  const blob = b.lastDownload();
  assert.strictEqual(blob.kind, 'likes',
                     `downloaded kind=${blob.kind}, expected likes`);
  assert.ok(blob.pages.length > 0, 'no pages captured');
  assert.ok(blob.pages.every(p => p.kind === 'likes'),
            'a bookmarks page leaked into the likes download');
});

await check('a run warns when pages from the other timeline arrive', async () => {
  const b = browser('bookmarks', new Map());
  b.deliver();
  await b.settle();

  b.test.click();
  b.navigate('/i/history/likes');
  b.deliver();
  await b.settle();

  assert.match(b.status(), /WARNING/, `status was: ${b.status()}`);
  assert.match(b.status(), /other timeline/);
});

await check('a run stops when the tab changes mid-capture', async () => {
  const b = browser('bookmarks', new Map());
  b.deliver();
  await b.settle();
  b.test.click();
  b.navigate('/i/history/likes');
  await b.settle();
  const blob = b.lastDownload();
  assert.strictEqual(blob.kind, 'bookmarks');
  assert.ok(blob.pages.every(p => p.kind === 'bookmarks'));
});

await check('the panel is not relabelled mid-capture', async () => {
  const b = browser('bookmarks', new Map());
  b.deliver();
  await b.settle();
  b.full.onclick();
  b.navigate('/i/history/likes');
  assert.strictEqual(b.full.textContent, 'Capture bookmarks',
                     'label changed while a run was in progress');
  await b.settle();
});

// ---------------------------------------------------------------------------
// reset known
// ---------------------------------------------------------------------------

await check('reset known clears the list when confirmed', async () => {
  const store = new Map([['ti.known.bookmarks', JSON.stringify(['1', '2', '3'])]]);
  const b = browser('bookmarks', store);
  b.reset.click();
  assert.strictEqual(store.has('ti.known.bookmarks'), false);
  assert.match(b.status(), /cleared/i);
});

await check('reset known keeps the list when cancelled', async () => {
  const store = new Map([['ti.known.bookmarks', JSON.stringify(['1', '2', '3'])]]);
  const b = browser('bookmarks', store);
  b.state.confirmAnswer = false;
  b.reset.click();
  assert.deepStrictEqual(b.knownIds(), ['1', '2', '3']);
});

await check('reset known follows the current tab, not the panel\'s origin', async () => {
  const store = new Map([
    ['ti.known.bookmarks', JSON.stringify(['1'])],
    ['ti.known.likes', JSON.stringify(['2'])]]);
  const b = browser('bookmarks', store);
  b.navigate('/i/history/likes');
  b.reset.click();
  assert.strictEqual(store.has('ti.known.likes'), false, 'should clear likes');
  assert.ok(store.has('ti.known.bookmarks'), 'bookmarks should be untouched');
});

await check('reset makes a would-be-early-stopping run capture everything', async () => {
  const store = new Map();

  const first = browser('bookmarks', store);
  first.deliver();
  await first.settle();
  first.full.click();
  await first.settle();

  const second = browser('bookmarks', store);
  second.reset.click();
  second.deliver();
  await second.settle();
  second.full.click();
  await second.settle();
  assert.strictEqual(contentPages(second.lastDownload()), second.totalPages());
});

// ---------------------------------------------------------------------------
// paths
// ---------------------------------------------------------------------------

for (const [path, expected] of [
  ['/i/history', 'bookmarks'],
  ['/i/history/', 'bookmarks'],
  ['/i/history/like', 'likes'],
  ['/i/history/likes', 'likes'],
  ['/i/history/likes/', 'likes'],
]) {
  await check(`path ${path} is recognised as ${expected}`, async () => {
    const b = browser(expected, new Map(), { path });
    assert.ok(b.panel, `no panel was added for ${path}`);
    assert.strictEqual(b.full.textContent, 'Capture ' + expected);
    assert.match(b.logs()[0], new RegExp(`kind=${expected}`));
  });
}

for (const path of ['/home', '/i/bookmarks', '/i/history/foo', '/']) {
  await check(`path ${path} gets no panel`, async () => {
    const b = browser('bookmarks', new Map(), { path });
    assert.strictEqual(b.panel, undefined, `a panel appeared on ${path}`);
    assert.match(b.logs()[0], /kind=null/);
  });
}

await check('the console line names the path and kind', async () => {
  const b = browser('likes', new Map(), { path: '/i/history/likes' });
  assert.strictEqual(b.logs()[0],
                     'Tweet Index: path=/i/history/likes kind=likes');
});

await check('the console line hints at the fix when there is no panel', async () => {
  const b = browser('bookmarks', new Map(), { path: '/home' });
  assert.match(b.logs()[0], /no panel/);
  assert.match(b.logs()[0], /i\/history/);
});

await check('navigation logs a line each time the path changes', async () => {
  const b = browser('bookmarks', new Map());
  b.navigate('/i/history/likes');
  b.navigate('/home');
  assert.strictEqual(b.logs().length, 3, b.logs().join(' | '));
  assert.match(b.logs()[1], /path=\/i\/history\/likes kind=likes/);
  assert.match(b.logs()[2], /kind=null/);
});

// ---------------------------------------------------------------------------

await check('buttons are locked while a run is in progress', async () => {
  const b = browser('bookmarks', new Map());
  b.deliver();
  await b.settle();
  b.full.onclick();
  assert.ok(b.full.disabled && b.test.disabled && b.reset.disabled,
            'buttons should be disabled during a run');
  await b.settle();
  assert.ok(!b.full.disabled && !b.test.disabled && !b.reset.disabled,
            'buttons should be re-enabled afterwards');
});

await check('clicking a button off-timeline does nothing but warn', async () => {
  const b = browser('bookmarks', new Map());
  const btn = b.test;
  b.navigate('/home');
  btn.click();
  await b.settle();
  assert.strictEqual(b.state.downloads.length, 0, 'nothing should download');
});

// ---------------------------------------------------------------------------

for (const [verdict, name, err] of results)
  console.log(`${verdict}  ${name}${err ? `\n      ${err}` : ''}`);
const failed = results.filter(r => r[0] === 'FAIL').length;
console.log(`\n${results.length - failed} passed, ${failed} failed`);
process.exit(failed ? 1 : 0);

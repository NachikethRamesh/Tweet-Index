// ==UserScript==
// @name         Tweet Index capture
// @namespace    tweet-index
// @version      1.5.0
// @description  Copy the bookmarks and likes responses X's own page receives.
// @match        https://x.com/*
// @run-at       document-start
// @inject-into  page
// @grant        none
// ==/UserScript==

// This script never asks X for anything and never acts on the account. It
// wraps fetch and XMLHttpRequest so that responses the page is already
// receiving can be copied, scrolls the timeline at a human pace, and saves
// what it saw to a file in Downloads. All parsing happens later, in Python.

(() => {
  const GQL = /\/i\/api\/graphql\/[^/]+\/([A-Za-z]+)/;   // any GraphQL op
  const CFG = { stopAfterKnown: 40, idleScrolls: 6, maxMinutes: 20,
                minDelay: 2000, jitter: 1000,
                testPages: 2 };            // used by the test button only

  let pages = [];                           // raw responses, saved verbatim
  const seen = { bookmarks: new Set(), likes: new Set() };  // insertion order
  const known = {};
  for (const k of ['bookmarks', 'likes'])
    known[k] = new Set(JSON.parse(localStorage.getItem('ti.known.' + k) || '[]'));
  const streak = { bookmarks: 0, likes: 0 };  // consecutive known ids, per timeline

  let running = null;        // the feed being captured, or null
  let foreign = 0;           // responses from the OTHER feed during a run
  let scroll = null;         // last known scroll position, for the status line
  let sawEmptyPage = false;  // X answered with a page holding no tweets

  // Which timeline is on screen, or null if this is some other page.
  //
  // X serves likes at /i/history/likes. The design document says
  // /i/history/like, so both spellings are accepted rather than betting on
  // either one; the likes check runs first because /i/history is a prefix of
  // both. Trailing slashes are ignored.
  function currentKind() {
    const path = location.pathname.replace(/\/+$/, '') || '/';
    if (path.startsWith('/i/history/like')) return 'likes';
    if (path === '/i/history') return 'bookmarks';
    return null;
  }

  // Timeline entries look like {entryId: "tweet-<id>", content: {...}}.
  // Walk the whole payload so path changes above entries don't matter.
  function entryIds(node, out) {
    if (Array.isArray(node)) { node.forEach(n => entryIds(n, out)); return; }
    if (!node || typeof node !== 'object') return;
    const m = typeof node.entryId === 'string' && node.entryId.match(/^tweet-(\d+)$/);
    if (m) { out.push(m[1]); return; }
    for (const v of Object.values(node)) entryIds(v, out);
  }

  // Classify by response shape, not operation name (verified 2026-09-29).
  function classify(json) {
    const d = json && json.data;
    if (d && d.bookmark_timeline_v2) return 'bookmarks';
    const t = d && d.user && d.user.result && d.user.result.timeline;
    const page = t && t.timeline && t.timeline.metadata
                 && t.timeline.metadata.scribeConfig && t.timeline.metadata.scribeConfig.page;
    return page === 'favorites' ? 'likes' : null;   // everything else is ignored
  }

  function onPayload(url, json) {
    const kind = classify(json);
    if (!kind) return;
    const op = (url.match(GQL) || [])[1];            // kept for provenance only
    pages.push({ kind, op, url, capturedAt: new Date().toISOString(), json });
    // A response from the other timeline during a run means the page changed
    // underneath us. It is kept out of the download and reported.
    if (running && kind !== running) foreign++;
    const ids = []; entryIds(json, ids);
    // A response for this feed carrying no tweets at all is X saying there is
    // nothing further down. That is the end of the timeline, as opposed to a
    // load that merely stalled.
    if (running === kind && ids.length === 0) sawEmptyPage = true;
    for (const id of ids) {
      if (seen[kind].has(id)) continue;
      seen[kind].add(id);
      streak[kind] = known[kind].has(id) ? streak[kind] + 1 : 0;
    }
    status();
  }

  const origFetch = window.fetch;                       // patch fetch
  window.fetch = async function (...args) {
    const res = await origFetch.apply(this, args);
    try {
      const url = typeof args[0] === 'string' ? args[0] : args[0].url;
      if (GQL.test(url)) res.clone().json().then(j => onPayload(url, j)).catch(() => {});
    } catch (_) {}
    return res;
  };
  const origOpen = XMLHttpRequest.prototype.open;       // patch XHR
  XMLHttpRequest.prototype.open = function (method, url, ...rest) {
    if (GQL.test(url)) this.addEventListener('load', () => {
      try { onPayload(url, JSON.parse(this.responseText)); } catch (_) {}
    });
    return origOpen.call(this, method, url, ...rest);
  };

  const sleep = ms => new Promise(r => setTimeout(r, ms));
  function pageCount(kind) { return pages.filter(p => p.kind === kind).length; }

  const STUCK_LIMIT = 3;        // scrolls that move nothing before giving up

  // Why did the loop end? The conditions are checked in the order that makes
  // the most useful sentence: a deliberate limit beats a symptom, and a
  // symptom beats "nothing new happened".
  //
  // `code` decides whether the known-ID list may be updated. Only two reasons
  // mean the run actually caught up with what we already had:
  //
  //   'known' - it met 40 tweets it had seen before, so everything newer is in
  //   'end'   - it reached the bottom of the timeline
  //
  // Every other reason means the run stopped somewhere in the middle. Recording
  // IDs then would be harmful: the next run starts at the newest tweet again,
  // would meet 40 known IDs within the first page or two, stop, and never reach
  // anything below where this run gave up. The gap would be permanent.
  function stopReason(kind, pageLimit, t0, idle, stuck) {
    if (currentKind() !== kind)
      return { code: 'tab', text: 'the tab changed to another page' };
    if (pageLimit && pageCount(kind) >= pageLimit)
      return { code: 'pages', text: `the ${pageLimit}-page test limit` };
    if (streak[kind] >= CFG.stopAfterKnown)
      return { code: 'known',
               text: `${CFG.stopAfterKnown} already-saved tweets in a row `
                     + '(everything newer has been captured)' };
    if (Date.now() - t0 >= CFG.maxMinutes * 60000)
      return { code: 'time', text: `the ${CFG.maxMinutes}-minute time cap` };
    if (stuck >= STUCK_LIMIT)
      return { code: 'stuck',
               text: `the page stopped scrolling (${stuck} scrolls moved nothing)` };
    if (sawEmptyPage)
      return { code: 'end', text: 'the end of the timeline' };
    if (idle >= CFG.idleScrolls)
      return { code: 'idle',
               text: `${CFG.idleScrolls} scrolls at the bottom with nothing new` };
    return { code: 'unknown', text: 'an unrecognised condition' };
  }

  // The run reached tweets it had already saved, so the newest IDs are a safe
  // starting point for next time.
  const CAUGHT_UP = ['known', 'end'];

  // --- scrolling ------------------------------------------------------------
  //
  // X loads the next page only when you are near the bottom of what is already
  // rendered. Twenty tweets with images are ten to fifteen screens tall, so
  // nudging down by one screen at a time never gets there: the run looks idle
  // and stops after the first page. Each step therefore jumps all the way to
  // the bottom.
  //
  // Which element scrolls is not guaranteed either. Usually it is the page,
  // but a single-page app can put the timeline in its own overflow container,
  // in which case scrolling the page does nothing at all. findScroller looks
  // for whichever element can actually move.

  let scroller = null;

  function findScroller() {
    const page = document.scrollingElement || document.documentElement;
    if (page && page.scrollHeight > page.clientHeight + 100) return page;
    let best = null;
    for (const el of document.querySelectorAll('div, main, section')) {
      if (el.scrollHeight <= el.clientHeight + 100) continue;
      const overflow = getComputedStyle(el).overflowY;
      if (overflow !== 'auto' && overflow !== 'scroll') continue;
      if (!best || el.scrollHeight > best.scrollHeight) best = el;
    }
    return best || page;
  }

  /** Where we are in the scrolling element, in pixels. */
  function position(el) {
    const top = el.scrollTop;
    const client = el.clientHeight;
    const height = el.scrollHeight;
    // "At the bottom" needs slack: heights are fractional and X keeps
    // appending a little as it renders.
    return { top, client, height, atBottom: top + client >= height - 4 };
  }

  function scrollToBottom(el) {
    const target = el.scrollHeight;
    el.scrollTop = target;
    // For the page itself, scrollTop alone is not reliable everywhere.
    if (el === (document.scrollingElement || document.documentElement))
      window.scrollTo(0, target);
  }

  // pageLimit is for the test run only; leave it undefined for a real capture.
  // A test run downloads a file exactly like a real one, but must NOT record
  // known ids: writing the 40 ids it saw would make the next full run see a
  // known streak of 40 at the top of the timeline and stop after two pages.
  async function capture(kind, pageLimit) {
    const isTest = Boolean(pageLimit);
    const t0 = Date.now(); let idle = 0, last = seen[kind].size;
    running = kind;
    foreign = 0;
    scroll = null;
    lastMessage = null;            // this run's progress replaces the old text

    scroller = findScroller();
    sawEmptyPage = false;
    let stuck = 0;                 // scrolls that did not move the page

    while (streak[kind] < CFG.stopAfterKnown && idle < CFG.idleScrolls
           && stuck < STUCK_LIMIT
           && Date.now() - t0 < CFG.maxMinutes * 60000
           && !(pageLimit && pageCount(kind) >= pageLimit)
           && currentKind() === kind) {   // stop if the page changed under us
      const before = position(scroller);
      scrollToBottom(scroller);
      await sleep(CFG.minDelay + Math.random() * CFG.jitter);
      const after = position(scroller);

      const gotTweets = seen[kind].size !== last;
      const moved = after.top > before.top + 1 || after.height > before.height + 1;

      if (gotTweets) {
        idle = 0;                  // new tweets arrived: keep going
        stuck = 0;
      } else if (after.atBottom) {
        idle++;                    // at the bottom with nothing new: the end
        stuck = 0;
      } else if (!moved) {
        // Nothing new, not at the bottom, and the page did not budge. Either
        // we are scrolling the wrong element or the app is wedged. Try to find
        // the right element again before giving up.
        stuck++;
        if (stuck === 2) {
          const other = findScroller();
          if (other !== scroller) { scroller = other; stuck = 0; }
        }
      } else {
        idle = 0;                  // still travelling down a long page
        stuck = 0;
      }

      last = seen[kind].size;
      scroll = after;              // for the status line
      status();
    }

    const why = stopReason(kind, pageLimit, t0, idle, stuck);

    // Only ever save pages from the feed being captured.
    const mine = pages.filter(p => p.kind === kind);
    const tweets = seen[kind].size;
    const strays = foreign;

    const blob = new Blob([JSON.stringify({ kind, pages: mine })],
                          { type: 'application/json' });
    const a = Object.assign(document.createElement('a'), {
      href: URL.createObjectURL(blob),
      download: `capture_${kind}_${new Date().toISOString().replace(/[:.]/g, '-')}.json` });
    document.body.appendChild(a); a.click(); a.remove();

    const caughtUp = CAUGHT_UP.includes(why.code);
    if (!isTest && caughtUp)
      localStorage.setItem('ti.known.' + kind,
        JSON.stringify([...seen[kind]].slice(0, 200)));  // newest first

    // Start the next run clean, so it cannot re-download this run's pages.
    pages = [];
    seen[kind].clear();
    streak[kind] = 0;
    running = null;
    foreign = 0;

    status(`Saved ${mine.length} page(s), ${tweets} tweets as ${kind}. `
           + `Stopped: ${why.text}. `
           + (isTest
              ? 'Known ids NOT changed (test run). '
              : caughtUp
                ? ''
                : 'Known ids NOT changed (run did not reach saved tweets). ')
           + (strays ? `WARNING: ignored ${strays} page(s) from the other `
                       + 'timeline - did the tab change mid-run? ' : '')
           + 'Check Downloads.');
  }

  function resetKnown(kind) {
    const n = known[kind].size;
    if (!confirm(`Forget the ${n} known ${kind} ids?\n\n`
                 + 'The next run will then behave like a first backfill: it '
                 + 'scrolls to the end of the timeline. Ingest skips anything '
                 + 'already stored, so the only cost is time.')) return;
    localStorage.removeItem('ti.known.' + kind);
    known[kind].clear();
    streak[kind] = 0;
    status(`Known ids cleared for ${kind} (was ${n}).`);
  }

  // --- panel ---------------------------------------------------------------

  let panel = null, buttons = null;

  // The last thing a finished run (or Reset known) had to say. It is kept so
  // that relabelling the panel afterwards does not wipe a warning off screen.
  let lastMessage = null;

  function status(message) {
    if (message) lastMessage = message;
    if (!panel) return;
    const kind = running || currentKind();
    const where = scroll
      ? `scroll ${Math.round(scroll.top)}/`
        + `${Math.round(scroll.height - scroll.client)}`
        + (scroll.atBottom ? ' (bottom)' : '')
      : 'scroll ?';
    panel.querySelector('.ti-status').textContent = message || lastMessage
      || (running
          ? `${running}: ${pageCount(running)} pages, ${seen[running].size} tweets, `
            + `${where}, known streak ${streak[running]}/${CFG.stopAfterKnown}`
          : `ready on ${kind} - ${pageCount(kind)} page(s) seen so far`);
  }

  // The feed is read here, when the button is clicked, not when the panel was
  // built. X switches between the History tabs without reloading, so a panel
  // built on one tab can still be on screen while another tab is showing.
  function start(pageLimit) {
    const kind = currentKind();
    if (!kind) {
      status('Not on a timeline. Open /i/history or /i/history/likes.');
      return;
    }
    lock(true);
    capture(kind, pageLimit).finally(() => { lock(false); relabel(); });
  }

  function lock(on) { for (const b of buttons) b.disabled = on; }

  function relabel() {
    if (!panel) return;
    const kind = currentKind();
    if (!kind) return;
    buttons[0].textContent = 'Capture ' + kind;
    buttons[2].onclick = () => resetKnown(kind);
    status();
  }

  function addPanel() {
    const kind = currentKind();
    if (!kind || panel) return;

    panel = document.createElement('div');
    Object.assign(panel.style, {
      position: 'fixed', top: '12px', right: '12px', zIndex: 99999,
      background: '#15202b', color: '#fff', padding: '10px 12px',
      borderRadius: '10px', font: '13px/1.4 system-ui, sans-serif',
      boxShadow: '0 2px 12px rgba(0,0,0,.4)', maxWidth: '260px' });

    const full = document.createElement('button');
    full.textContent = 'Capture ' + kind;
    full.onclick = () => start(null);

    const test = document.createElement('button');
    test.textContent = `Test (${CFG.testPages} pages)`;
    test.title = 'Short run for checking the setup before a full capture. '
                 + 'Downloads a file but does not record known ids.';
    test.onclick = () => start(CFG.testPages);

    const reset = document.createElement('button');
    reset.textContent = 'Reset known';
    reset.title = 'Forget which tweets earlier runs saw, so the next run '
                  + 'scrolls the whole timeline again';
    reset.onclick = () => resetKnown(currentKind());

    buttons = [full, test, reset];
    for (const b of buttons)
      Object.assign(b.style, { display: 'block', width: '100%',
        margin: '0 0 6px', padding: '6px 8px', cursor: 'pointer' });

    const line = document.createElement('div');
    line.className = 'ti-status';
    Object.assign(line.style, { opacity: '.85', wordBreak: 'break-word' });

    panel.append(full, test, reset, line);
    document.body.appendChild(panel);
    status();
  }

  function removePanel() {
    if (!panel) return;
    panel.remove();
    panel = null;
    buttons = null;
  }

  // --- navigation ----------------------------------------------------------

  // X is a single-page app: moving between the History tabs changes the URL
  // without reloading, so the panel has to be rebuilt to match.
  let lastPath = null;
  function onNavigate() {
    if (location.pathname === lastPath) return;
    lastPath = location.pathname;
    const kind = currentKind();
    console.log(`Tweet Index: path=${location.pathname} kind=${kind}`
                + (kind ? '' : ' (no panel; open /i/history or '
                              + '/i/history/likes and reload)'));
    if (running) return;          // never change the panel mid-capture
    if (!kind) { removePanel(); return; }
    if (panel) relabel(); else addPanel();
  }

  for (const name of ['pushState', 'replaceState']) {
    const orig = history[name];
    history[name] = function (...args) {
      const out = orig.apply(this, args);
      try { onNavigate(); } catch (_) {}
      return out;
    };
  }
  window.addEventListener('popstate', onNavigate);
  setInterval(onNavigate, 1000);          // safety net for any route we miss

  document.addEventListener('DOMContentLoaded', onNavigate);
  if (document.readyState !== 'loading') onNavigate();   // script ran late
})();

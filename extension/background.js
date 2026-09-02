/*
 * SONIX service worker.
 *
 * Owns the capture lifecycle. It never touches audio itself — Manifest V3
 * service workers have no DOM and no AudioContext, so all real work happens in
 * the offscreen document. This file:
 *
 *   1. mints a tab-capture stream id for the Meet tab (requires a user gesture,
 *      which is the popup button click)
 *   2. makes sure exactly one offscreen document exists
 *   3. hands the stream id over and relays state back to the popup and to the
 *      on-page overlay
 */

const OFFSCREEN_PATH = "offscreen.html";

let state = {
  capturing: false,
  tabId: null,
  callId: null,
  pairingCode: null,
  callState: null,          // consent_pending | listening | scoring | ended
  scoringAvailable: false,
  lastScore: null,
  windows: 0,
  error: null,
};

/* ------------------------------------------------------------------ */
/* Offscreen document                                                  */
/* ------------------------------------------------------------------ */

async function hasOffscreen() {
  const contexts = await chrome.runtime.getContexts({
    contextTypes: ["OFFSCREEN_DOCUMENT"],
  });
  return contexts.length > 0;
}

async function ensureOffscreen() {
  if (await hasOffscreen()) return;
  await chrome.offscreen.createDocument({
    url: OFFSCREEN_PATH,
    reasons: ["USER_MEDIA"],
    justification:
      "Capture tab audio and stream it to the local SONIX detection server.",
  });
}

/* ------------------------------------------------------------------ */
/* Start / stop                                                        */
/* ------------------------------------------------------------------ */

async function startCapture(serverUrl) {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  if (!tab) throw new Error("No active tab.");
  if (!/^https:\/\/meet\.google\.com\//.test(tab.url || "")) {
    throw new Error("Open a Google Meet call in this tab first.");
  }

  // Must be called from a user gesture. The popup button provides it.
  const streamId = await chrome.tabCapture.getMediaStreamId({
    targetTabId: tab.id,
  });

  await ensureOffscreen();

  state = { ...state, capturing: true, tabId: tab.id, error: null, windows: 0 };

  chrome.runtime.sendMessage({
    target: "offscreen",
    type: "start",
    streamId,
    serverUrl,
    caller: "google-meet",
  });

  updateBadge();
}

async function stopCapture() {
  chrome.runtime.sendMessage({ target: "offscreen", type: "stop" });
  state = {
    ...state,
    capturing: false,
    callId: null,
    pairingCode: null,
    callState: null,
    lastScore: null,
  };
  updateBadge();
  pushToOverlay();
}

/* ------------------------------------------------------------------ */
/* UI feedback                                                         */
/* ------------------------------------------------------------------ */

function bandFor(score) {
  // SONIX tokens - keep in step with popup.css / demo/theme.py.
  if (score === null || score === undefined) return { label: "—", colour: "#7e8d8d" };
  if (score >= 0.65) return { label: "RED", colour: "#f0685f" };
  if (score >= 0.35) return { label: "AMBER", colour: "#e0a92b" };
  return { label: "GREEN", colour: "#39c26a" };
}

function updateBadge() {
  if (!state.capturing) {
    chrome.action.setBadgeText({ text: "" });
    return;
  }
  if (!state.scoringAvailable) {
    chrome.action.setBadgeText({ text: "REC" });
    chrome.action.setBadgeBackgroundColor({ color: "#0b7c86" });
    return;
  }
  const b = bandFor(state.lastScore);
  chrome.action.setBadgeText({ text: b.label[0] });
  chrome.action.setBadgeBackgroundColor({ color: b.colour });
}

function pushToOverlay() {
  if (state.tabId === null) return;
  chrome.tabs
    .sendMessage(state.tabId, { type: "sonix:state", state })
    .catch(() => {
      /* content script not injected on this page — ignore */
    });
}

/* ------------------------------------------------------------------ */
/* Message routing                                                     */
/* ------------------------------------------------------------------ */

chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  // From the popup
  if (msg.type === "popup:start") {
    startCapture(msg.serverUrl)
      .then(() => sendResponse({ ok: true }))
      .catch((e) => {
        state.error = e.message;
        sendResponse({ ok: false, error: e.message });
      });
    return true;
  }
  if (msg.type === "popup:stop") {
    stopCapture().then(() => sendResponse({ ok: true }));
    return true;
  }
  if (msg.type === "popup:state") {
    sendResponse(state);
    return false;
  }

  // From the offscreen document
  if (msg.target === "background") {
    if (msg.type === "call_started") {
      state.callId = msg.callId;
      state.pairingCode = msg.pairingCode;
      state.callState = "consent_pending";
    } else if (msg.type === "call_state") {
      state.callState = msg.state;
    } else if (msg.type === "score") {
      state.lastScore = msg.score;
      state.windows = msg.windows ?? state.windows;
    } else if (msg.type === "scoring_available") {
      state.scoringAvailable = msg.value;
    } else if (msg.type === "error") {
      state.error = msg.message;
      state.capturing = false;
    } else if (msg.type === "closed") {
      state.capturing = false;
    }
    updateBadge();
    pushToOverlay();
  }
  return false;
});

// Re-send state when the Meet tab finishes loading, so the overlay reappears.
chrome.tabs.onUpdated.addListener((tabId, info) => {
  if (tabId === state.tabId && info.status === "complete") pushToOverlay();
});

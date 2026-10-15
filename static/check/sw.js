// Service worker of the offline message checker (static/check/).
//
// Saves every file of the page — including the model — on the phone the
// first time the page is opened, then serves them from there: the page
// opens instantly, even when the website is asleep or there is no internet.
//
// VERSION is a hash of all the files below. When anything changes (a new
// model, a fixed page) the build script rewrites it, the browser sees a new
// sw.js, installs the new files and deletes the old ones.

// BEGIN GENERATED (scripts/build_web_checker.py)
const VERSION = "ed80b3f3b57b1d9e";
const ASSETS = [
  "app.css",
  "app.js",
  "icon.png",
  "index.html",
  "lib/detector.js",
  "lib/ensemble.js",
  "lib/ensemble_model.json",
  "lib/i18n.js",
  "lib/mistakes.js",
  "lib/normalize.js",
  "lib/scorer.js",
  "manifest.json"
];
// END GENERATED

const CACHE = "afd-check-" + VERSION;

self.addEventListener("install", (event) => {
  event.waitUntil(
    caches.open(CACHE)
      .then((cache) => cache.addAll(ASSETS.map((a) => new Request(a, { cache: "reload" }))))
      .then(() => self.skipWaiting())
  );
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches.keys()
      .then((keys) => Promise.all(keys.filter((k) => k.startsWith("afd-check-") && k !== CACHE).map((k) => caches.delete(k))))
      .then(() => self.clients.claim())
  );
});

// Cache first: the page must work offline and open fast. The query string
// (?lang=ru) and the #text=... fragment never change which file is meant.
self.addEventListener("fetch", (event) => {
  const req = event.request;
  if (req.method !== "GET" || new URL(req.url).origin !== self.location.origin) return;
  event.respondWith(
    caches.open(CACHE).then((cache) =>
      cache.match(req, { ignoreSearch: true }).then((hit) => {
        if (hit) return hit;
        if (req.mode === "navigate") {
          return cache.match("./index.html").then((page) => page || fetch(req));
        }
        return fetch(req);
      })
    )
  );
});

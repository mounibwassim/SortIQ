const CACHE_NAME = 'sortiq-cache-v1';

const APP_SHELL = [
  '/',
  '/index.html',
  '/manifest.json',
  '/icon-192.png',
  '/icon-512.png'
];

// Install Event - cache app shell
self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => {
      console.log('Opened cache');
      return cache.addAll(APP_SHELL);
    })
  );
  self.skipWaiting();
});

// Activate Event - clean up old caches
self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((cacheNames) => {
      return Promise.all(
        cacheNames.map((cacheName) => {
          if (cacheName !== CACHE_NAME) {
            return caches.delete(cacheName);
          }
        })
      );
    })
  );
  self.clients.claim();
});

// Fetch Event - network first, fallback to cache, then offline message
self.addEventListener('fetch', (event) => {
  // We only handle GET requests and http/https schemes
  if (event.request.method !== 'GET') return;
  if (!(event.request.url.startsWith('http://') || event.request.url.startsWith('https://'))) return;

  event.respondWith(
    fetch(event.request)
      .then((networkResponse) => {
        // If it's a valid response, clone it and cache it dynamically
        if (networkResponse && networkResponse.status === 200 && networkResponse.type === 'basic') {
          const responseToCache = networkResponse.clone();
          caches.open(CACHE_NAME).then((cache) => {
            cache.put(event.request, responseToCache);
          });
        }
        return networkResponse;
      })
      .catch(async () => {
        // Network failed (offline), try the cache
        const cachedResponse = await caches.match(event.request);
        if (cachedResponse) {
          return cachedResponse;
        }

        // If it's an API request, return JSON generic offline
        if (event.request.url.includes('/api/') || event.request.url.includes(':8000')) {
          return new Response(JSON.stringify({ error: 'You are offline. Please check your internet connection.' }), {
            headers: { 'Content-Type': 'application/json' },
            status: 503
          });
        }

        // Otherwise return a generic offline HTML page or let it fail
        return new Response(
          '<html><head><title>SortIQ Offline</title><style>body{font-family:sans-serif;text-align:center;padding:50px;color:#333;} h1{color:#22c55e;}</style></head><body><h1>You are offline</h1><p>Please check your network connection and try again.</p></body></html>',
          {
            headers: { 'Content-Type': 'text/html' }
          }
        );
      })
  );
});

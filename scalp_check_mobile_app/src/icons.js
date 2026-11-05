// Line icons used across the app (1.75 stroke, same as the design kit).
const P = {
  home: '<path d="M4 11l8-7 8 7v9a1 1 0 0 1-1 1h-4v-6H9v6H5a1 1 0 0 1-1-1z"/>',
  cam: '<path d="M4 8h3l2-3h6l2 3h3v11H4z"/><circle cx="12" cy="13" r="3.5"/>',
  book: '<path d="M5 4h10a3 3 0 0 1 3 3v13H8a3 3 0 0 1-3-3z"/><path d="M5 17a3 3 0 0 1 3-3h10"/>',
  info: '<circle cx="12" cy="12" r="9"/><path d="M12 11v6M12 7.5v.1"/>',
  back: '<path d="M15 5l-7 7 7 7"/>',
  sun: '<circle cx="12" cy="12" r="4"/><path d="M12 3v2M12 19v2M3 12h2M19 12h2M5.6 5.6L7 7M17 17l1.4 1.4M5.6 18.4L7 17M17 7l1.4-1.4"/>',
  scan: '<path d="M4 9V4h5M15 4h5v5M20 15v5h-5M9 20H4v-5"/>',
  lock: '<rect x="5" y="11" width="14" height="9" rx="2"/><path d="M8 11V8a4 4 0 0 1 8 0v3"/>',
  next: '<path d="M9 5l7 7-7 7"/>',
  down: '<path d="M6 9l6 6 6-6"/>',
  user: '<circle cx="12" cy="8" r="4"/><path d="M4 21c1-5 5-7 8-7s7 2 8 7"/>',
};

export function icon(name, size = 24, color = '#0F6E7A') {
  return `<svg width="${size}" height="${size}" viewBox="0 0 24 24" fill="none" stroke="${color}" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${P[name]}</svg>`;
}

export const CHECK = '<svg width="16" height="16" viewBox="0 0 16 16" aria-hidden="true"><circle cx="8" cy="8" r="8" fill="#1B8A7A"/><path d="M4.5 8.2l2.3 2.3 4.5-4.8" stroke="#fff" stroke-width="1.75" fill="none" stroke-linecap="round" stroke-linejoin="round"/></svg>';
export const WARN = '<svg width="16" height="16" viewBox="0 0 16 16" aria-hidden="true"><circle cx="8" cy="8" r="8" fill="#E0A100"/><path d="M8 4.5v4.2M8 11v.1" stroke="#3D2A00" stroke-width="1.75" stroke-linecap="round"/></svg>';

export function logo(bg = '#fff', fg = '#0F6E7A', size = 30) {
  return `<svg width="${size}" height="${size}" viewBox="0 0 32 32" aria-hidden="true"><circle cx="16" cy="16" r="15" fill="${bg}"/><path d="M10 22c0-6 3-9 6-12M16 23c0-5 1-8 3-11M22 21c0-4 0-6-1-9" stroke="${fg}" stroke-width="1.75" fill="none" stroke-linecap="round"/></svg>`;
}

export const HAIR_MARK = '<svg class="hair-mark" width="150" height="150" viewBox="0 0 32 32" fill="none" stroke="#fff" stroke-opacity=".12" stroke-width="1.2" stroke-linecap="round" aria-hidden="true"><path d="M10 22c0-6 3-9 6-12M16 23c0-5 1-8 3-11M22 21c0-4 0-6-1-9"/></svg>';

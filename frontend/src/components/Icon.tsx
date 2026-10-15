type IconName =
  | 'lab'
  | 'systems'
  | 'experiments'
  | 'evidence'
  | 'activity'
  | 'arrow'
  | 'external'
  | 'refresh'
  | 'copy'
  | 'check'
  | 'alert'

const paths: Record<IconName, React.ReactNode> = {
  lab: <><path d="M8 3v5l-4.5 8.1A2 2 0 0 0 5.2 19h13.6a2 2 0 0 0 1.7-2.9L16 8V3"/><path d="M7 13h10M6 3h12"/></>,
  systems: <><rect x="3" y="4" width="7" height="7" rx="1"/><rect x="14" y="4" width="7" height="7" rx="1"/><rect x="3" y="15" width="7" height="5" rx="1"/><rect x="14" y="15" width="7" height="5" rx="1"/></>,
  experiments: <><path d="M9 3h6M10 3v6l-5 8a2 2 0 0 0 1.7 3h10.6a2 2 0 0 0 1.7-3l-5-8V3"/><path d="M8 14h8"/></>,
  evidence: <><path d="M4 5a2 2 0 0 1 2-2h12a2 2 0 0 1 2 2v14a1 1 0 0 1-1.5.9L12 16l-6.5 3.9A1 1 0 0 1 4 19Z"/><path d="M8 7h8M8 11h6"/></>,
  activity: <><path d="M3 12h4l2-6 4 12 2-6h6"/></>,
  arrow: <><path d="M5 12h14M14 7l5 5-5 5"/></>,
  external: <><path d="M14 4h6v6M20 4l-9 9"/><path d="M18 13v6a1 1 0 0 1-1 1H5a1 1 0 0 1-1-1V7a1 1 0 0 1 1-1h6"/></>,
  refresh: <><path d="M20 11a8 8 0 1 0-2.3 5.7"/><path d="M20 4v7h-7"/></>,
  copy: <><rect x="8" y="8" width="11" height="11" rx="2"/><path d="M16 8V5a2 2 0 0 0-2-2H5a2 2 0 0 0-2 2v9a2 2 0 0 0 2 2h3"/></>,
  check: <path d="m5 12 4 4L19 6"/>,
  alert: <><path d="M12 9v4M12 17h.01"/><path d="M10.3 3.7 2.6 17a2 2 0 0 0 1.7 3h15.4a2 2 0 0 0 1.7-3L13.7 3.7a2 2 0 0 0-3.4 0Z"/></>,
}

export function Icon({ name, size = 18 }: { name: IconName; size?: number }) {
  return <svg aria-hidden="true" width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round">{paths[name]}</svg>
}

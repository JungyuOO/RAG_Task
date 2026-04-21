export type AppRoute = "workspaces" | "connections" | "models" | "overview" | "resources" | "library" | "chat" | "actions" | "scm";

const ROUTE_PATHS: Record<AppRoute, string> = {
  workspaces: "/workspaces",
  connections: "/connections",
  models: "/models",
  overview: "/overview",
  resources: "/resources",
  library: "/library",
  chat: "/chat",
  actions: "/actions",
  scm: "/scm",
};

const LEGACY_PATHS: Record<string, AppRoute> = {
  "/": "connections",
  "/workspaces": "workspaces",
  "/connection": "connections",
  "/connections": "connections",
  "/models": "models",
  "/dashboard": "overview",
  "/overview": "overview",
  "/resources": "resources",
  "/library": "library",
  "/chat": "chat",
  "/actions": "actions",
  "/scm": "scm",
};

function normalizePathname(pathname: string): string {
  const value = String(pathname || "").trim();
  if (!value || value === "/") return "/";
  return value.endsWith("/") ? value.slice(0, -1) : value;
}

export function readRouteStateFromLocation(locationLike: Pick<Location, "pathname" | "search">): {
  route: AppRoute;
  chatSessionId: string;
} {
  const route = LEGACY_PATHS[normalizePathname(locationLike.pathname)] ?? "connections";
  const search = new URLSearchParams(locationLike.search);
  return {
    route,
    chatSessionId: route === "chat" ? search.get("session")?.trim() ?? "" : "",
  };
}

export function getCanonicalUrl(route: AppRoute, options?: { chatSessionId?: string }): string {
  const base = ROUTE_PATHS[route];
  if (route !== "chat") return base;
  const sessionId = options?.chatSessionId?.trim() ?? "";
  if (!sessionId) return base;
  const search = new URLSearchParams({ session: sessionId });
  return `${base}?${search.toString()}`;
}

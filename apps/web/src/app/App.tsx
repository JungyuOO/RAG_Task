import { Suspense, lazy, useEffect, useMemo, useState } from "react";

import { useOcpConnection } from "@/domains/connection/useOcpConnection";
import { ChatSessionRail } from "@/domains/chat/ChatSessionRail";
import { createChatSessionRecord, type ChatSessionRecord } from "@/domains/chat/types";
import type { WorkspaceRecord } from "@/domains/workspaces/types";
import { listWorkspaces } from "@/domains/workspaces/workspacesApi";
import { AppShell } from "@/shared/layout/AppShell";
import { getCanonicalUrl, readRouteStateFromLocation, type AppRoute } from "./routes";
const CHAT_SESSIONS_STORAGE_KEY = "rag-task.chat.sessions";
const ACTIVE_WORKSPACE_STORAGE_KEY = "rag-task.workspace.active";
const WorkspacesPage = lazy(() => import("../pages/WorkspacesPage").then((module) => ({ default: module.WorkspacesPage })));
const ConnectionPage = lazy(() => import("../pages/ConnectionPage").then((module) => ({ default: module.ConnectionPage })));
const ModelsPage = lazy(() => import("../pages/ModelsPage").then((module) => ({ default: module.ModelsPage })));
const DashboardPage = lazy(() => import("../pages/DashboardPage").then((module) => ({ default: module.DashboardPage })));
const ResourcesPage = lazy(() => import("../pages/ResourcesPage").then((module) => ({ default: module.ResourcesPage })));
const LibraryPage = lazy(() => import("../pages/LibraryPage").then((module) => ({ default: module.LibraryPage })));
const ChatPage = lazy(() => import("../pages/ChatPage").then((module) => ({ default: module.ChatPage })));
const ActionsPage = lazy(() => import("../pages/ActionsPage").then((module) => ({ default: module.ActionsPage })));

type GlobalLoadingState = {
  active: boolean;
  title: string;
  detail?: string;
};

function loadChatSessions(): ChatSessionRecord[] {
  if (typeof window === "undefined") return [createChatSessionRecord()];
  try {
    const raw = window.localStorage.getItem(CHAT_SESSIONS_STORAGE_KEY);
    if (!raw) return [createChatSessionRecord()];
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed) || parsed.length === 0) return [createChatSessionRecord()];
    return parsed as ChatSessionRecord[];
  } catch {
    return [createChatSessionRecord()];
  }
}

export function App() {
  const initialRouteState =
    typeof window !== "undefined"
      ? readRouteStateFromLocation(window.location)
      : { route: "connections" as AppRoute, chatSessionId: "" };
  const [selectedWorkspaceId, setSelectedWorkspaceId] = useState<string>(() => {
    if (typeof window === "undefined") return "";
    return window.localStorage.getItem(ACTIVE_WORKSPACE_STORAGE_KEY) ?? "";
  });
  const connectionController = useOcpConnection(selectedWorkspaceId);
  const [route, setRoute] = useState<AppRoute>(initialRouteState.route);
  const [chatSessions, setChatSessions] = useState<ChatSessionRecord[]>(() => loadChatSessions());
  const [activeChatSessionId, setActiveChatSessionId] = useState<string>(initialRouteState.chatSessionId);
  const [workspaceSnapshot, setWorkspaceSnapshot] = useState<WorkspaceRecord[]>([]);
  const [pageLoadingState, setPageLoadingState] = useState<GlobalLoadingState>({
    active: false,
    title: "",
  });

  useEffect(() => {
    window.localStorage.setItem(CHAT_SESSIONS_STORAGE_KEY, JSON.stringify(chatSessions));
  }, [chatSessions]);

  useEffect(() => {
    if (typeof window === "undefined") return;
    window.localStorage.setItem(ACTIVE_WORKSPACE_STORAGE_KEY, selectedWorkspaceId);
  }, [selectedWorkspaceId]);

  useEffect(() => {
    if (!activeChatSessionId && chatSessions[0]) {
      setActiveChatSessionId(chatSessions[0].id);
      return;
    }
    if (!chatSessions.some((session) => session.id === activeChatSessionId)) {
      setActiveChatSessionId(chatSessions[0]?.id ?? createChatSessionRecord().id);
    }
  }, [activeChatSessionId, chatSessions]);

  useEffect(() => {
    async function run() {
      try {
        const next = await listWorkspaces();
        setWorkspaceSnapshot(next);
        if (!selectedWorkspaceId && next[0]) {
          setSelectedWorkspaceId(next[0].workspaceId);
        }
      } catch {
        setWorkspaceSnapshot([]);
      }
    }
    void run();
  }, [selectedWorkspaceId]);

  useEffect(() => {
    if (typeof window === "undefined") return;
    const handlePopState = () => {
      const next = readRouteStateFromLocation(window.location);
      setRoute(next.route);
      if (next.chatSessionId) {
        setActiveChatSessionId(next.chatSessionId);
      }
    };
    window.addEventListener("popstate", handlePopState);
    return () => window.removeEventListener("popstate", handlePopState);
  }, []);

  const activeChatSession = useMemo(
    () => chatSessions.find((session) => session.id === activeChatSessionId) ?? chatSessions[0] ?? createChatSessionRecord(),
    [activeChatSessionId, chatSessions],
  );
  const selectedWorkspace = useMemo(
    () => workspaceSnapshot.find((item) => item.workspaceId === selectedWorkspaceId) ?? null,
    [selectedWorkspaceId, workspaceSnapshot],
  );

  function navigate(nextRoute: AppRoute, options?: { chatSessionId?: string; replace?: boolean }) {
    const nextUrl = getCanonicalUrl(nextRoute, {
      chatSessionId: options?.chatSessionId ?? (nextRoute === "chat" ? activeChatSessionId : ""),
    });
    setRoute(nextRoute);
    if (typeof window === "undefined") return;
    const currentUrl = `${window.location.pathname}${window.location.search}`;
    if (currentUrl === nextUrl) return;
    if (options?.replace) {
      window.history.replaceState(null, "", nextUrl);
      return;
    }
    window.history.pushState(null, "", nextUrl);
  }

  useEffect(() => {
    if (typeof window === "undefined") return;
    const nextUrl = getCanonicalUrl(route, {
      chatSessionId: route === "chat" ? activeChatSessionId : "",
    });
    const currentUrl = `${window.location.pathname}${window.location.search}`;
    if (currentUrl !== nextUrl) {
      window.history.replaceState(null, "", nextUrl);
    }
  }, [activeChatSessionId, route]);

  function createChatSession() {
    const nextSession = createChatSessionRecord();
    setChatSessions((current) => [nextSession, ...current]);
    setActiveChatSessionId(nextSession.id);
    navigate("chat", { chatSessionId: nextSession.id });
  }

  function selectChatSession(sessionId: string) {
    setActiveChatSessionId(sessionId);
    navigate("chat", { chatSessionId: sessionId });
  }

  function removeChatSession(sessionId: string) {
    setChatSessions((current) => {
      const next = current.filter((session) => session.id !== sessionId);
      return next.length > 0 ? next : [createChatSessionRecord()];
    });
  }

  function updateChatSession(sessionId: string, updater: (current: ChatSessionRecord) => ChatSessionRecord) {
    setChatSessions((current) => current.map((session) => (session.id === sessionId ? updater(session) : session)));
  }

  const shellLoadingState = useMemo<GlobalLoadingState>(() => {
    if (connectionController.isBusy) {
      const title =
        connectionController.submitState === "connecting"
          ? "클러스터 연결 중"
          : connectionController.submitState === "testing"
            ? "연결 상태 검증 중"
            : connectionController.submitState === "disconnecting"
              ? "세션 정리 중"
              : "처리 중";
      return { active: true, title, detail: connectionController.message };
    }
    return pageLoadingState;
  }, [connectionController.isBusy, connectionController.message, connectionController.submitState, pageLoadingState]);

  const page = useMemo(() => {
    switch (route) {
      case "workspaces":
        return (
          <WorkspacesPage
            selectedWorkspaceId={selectedWorkspaceId}
            onSelectWorkspace={setSelectedWorkspaceId}
            onLoadingChange={setPageLoadingState}
          />
        );
      case "chat":
        return <ChatPage controller={connectionController} session={activeChatSession} updateSession={updateChatSession} />;
      case "models":
        return <ModelsPage selectedWorkspace={selectedWorkspace} onLoadingChange={setPageLoadingState} />;
      case "overview":
        return <DashboardPage controller={connectionController} onLoadingChange={setPageLoadingState} />;
      case "resources":
        return <ResourcesPage controller={connectionController} onLoadingChange={setPageLoadingState} />;
      case "library":
        return <LibraryPage controller={connectionController} onLoadingChange={setPageLoadingState} />;
      case "actions":
        return <ActionsPage controller={connectionController} />;
      case "connections":
      default:
        return <ConnectionPage controller={connectionController} />;
    }
  }, [activeChatSession, connectionController, route, selectedWorkspace, selectedWorkspaceId]);

  const railContent = useMemo(() => {
    if (route !== "chat") return null;
    return (
      <ChatSessionRail
        sessions={chatSessions}
        activeSessionId={activeChatSession.id}
        onSelect={selectChatSession}
        onCreate={createChatSession}
        onRemove={removeChatSession}
      />
    );
  }, [activeChatSession.id, chatSessions, route]);

  return (
    <AppShell activeRoute={route} onNavigate={navigate} profile={connectionController.profile} testResult={connectionController.testResult} schedulerStatus={connectionController.schedulerStatus} message={connectionController.message} railContent={railContent} loadingState={shellLoadingState}>
      <Suspense fallback={<div className="rounded-xl border border-border/70 bg-background/50 px-4 py-3 text-sm text-muted-foreground">페이지를 불러오는 중입니다.</div>}>
        {page}
      </Suspense>
    </AppShell>
  );
}



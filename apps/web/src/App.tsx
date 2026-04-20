import { Suspense, lazy, useEffect, useMemo, useState } from "react";

import { useOcpConnection } from "@/features/connection/hooks/useOcpConnection";
import { ChatSessionRail } from "@/features/chat/components/ChatSessionRail";
import { createChatSessionRecord, type ChatSessionRecord } from "@/features/chat/types";
import { AppShell, type AppRoute } from "@/components/layout/AppShell";
const CHAT_SESSIONS_STORAGE_KEY = "rag-task.chat.sessions";
const ConnectionPage = lazy(() => import("./pages/ConnectionPage").then((module) => ({ default: module.ConnectionPage })));
const DashboardPage = lazy(() => import("./pages/DashboardPage").then((module) => ({ default: module.DashboardPage })));
const ResourcesPage = lazy(() => import("./pages/ResourcesPage").then((module) => ({ default: module.ResourcesPage })));
const LibraryPage = lazy(() => import("./pages/LibraryPage").then((module) => ({ default: module.LibraryPage })));
const ChatPage = lazy(() => import("./pages/ChatPage").then((module) => ({ default: module.ChatPage })));
const ActionsPage = lazy(() => import("./pages/ActionsPage").then((module) => ({ default: module.ActionsPage })));

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
  const connectionController = useOcpConnection();
  const [route, setRoute] = useState<AppRoute>("connection");
  const [chatSessions, setChatSessions] = useState<ChatSessionRecord[]>(() => loadChatSessions());
  const [activeChatSessionId, setActiveChatSessionId] = useState<string>("");
  const [pageLoadingState, setPageLoadingState] = useState<GlobalLoadingState>({
    active: false,
    title: "",
  });

  useEffect(() => {
    window.localStorage.setItem(CHAT_SESSIONS_STORAGE_KEY, JSON.stringify(chatSessions));
  }, [chatSessions]);

  useEffect(() => {
    if (!activeChatSessionId && chatSessions[0]) {
      setActiveChatSessionId(chatSessions[0].id);
      return;
    }
    if (!chatSessions.some((session) => session.id === activeChatSessionId)) {
      setActiveChatSessionId(chatSessions[0]?.id ?? createChatSessionRecord().id);
    }
  }, [activeChatSessionId, chatSessions]);

  const activeChatSession = useMemo(
    () => chatSessions.find((session) => session.id === activeChatSessionId) ?? chatSessions[0] ?? createChatSessionRecord(),
    [activeChatSessionId, chatSessions],
  );

  function createChatSession() {
    const nextSession = createChatSessionRecord();
    setChatSessions((current) => [nextSession, ...current]);
    setActiveChatSessionId(nextSession.id);
    setRoute("chat");
  }

  function selectChatSession(sessionId: string) {
    setActiveChatSessionId(sessionId);
    setRoute("chat");
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
      case "chat":
        return <ChatPage controller={connectionController} session={activeChatSession} updateSession={updateChatSession} />;
      case "dashboard":
        return <DashboardPage controller={connectionController} onLoadingChange={setPageLoadingState} />;
      case "resources":
        return <ResourcesPage controller={connectionController} onLoadingChange={setPageLoadingState} />;
      case "library":
        return <LibraryPage controller={connectionController} onLoadingChange={setPageLoadingState} />;
      case "actions":
        return <ActionsPage controller={connectionController} />;
      case "connection":
      default:
        return <ConnectionPage controller={connectionController} />;
    }
  }, [activeChatSession, connectionController, route]);

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
    <AppShell activeRoute={route} onNavigate={setRoute} profile={connectionController.profile} testResult={connectionController.testResult} schedulerStatus={connectionController.schedulerStatus} message={connectionController.message} railContent={railContent} loadingState={shellLoadingState}>
      <Suspense fallback={<div className="rounded-xl border border-border/70 bg-background/50 px-4 py-3 text-sm text-muted-foreground">페이지를 불러오는 중입니다.</div>}>
        {page}
      </Suspense>
    </AppShell>
  );
}

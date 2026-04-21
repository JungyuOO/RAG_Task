import { useEffect, useState } from "react";

import type {
  OcpConnectionProfile,
  OcpConnectionRequest,
  OcpConnectionTestResult,
  OcpLeaseSchedulerStatus,
  SavedOcpConnectionProfile,
} from "@/domains/connection/types";
import {
  connectOcp,
  disconnectOcp,
  getOcpConnectionStatus,
  getOcpLeaseSchedulerStatus,
  refreshOcpConnectionLease,
  testOcpConnection,
} from "./ocpConnectionApi";

const PROFILE_STORAGE_KEY = "rag-task.ocp.connection.profile";
const TEST_RESULT_STORAGE_KEY = "rag-task.ocp.connection.test-result";
const ACTIVE_SAVED_PROFILE_STORAGE_KEY = "rag-task.ocp.connection.saved-profile.active";
const SAVED_PROFILES_STORAGE_KEY = "rag-task.ocp.connection.saved-profiles";

type SubmitState = "idle" | "connecting" | "testing" | "disconnecting";

export type OcpConnectionSubmitState = SubmitState;

export interface OcpConnectionController {
  submitState: SubmitState;
  isBusy: boolean;
  message: string;
  profile: OcpConnectionProfile | null;
  testResult: OcpConnectionTestResult | null;
  schedulerStatus: OcpLeaseSchedulerStatus | null;
  savedProfiles: SavedOcpConnectionProfile[];
  activeSavedProfileId: string | null;
  createConnection: (request: OcpConnectionRequest) => Promise<OcpConnectionProfile | null>;
  connectSavedProfile: (savedProfileId: string) => Promise<OcpConnectionProfile | null>;
  removeSavedProfile: (savedProfileId: string) => void;
  runConnectionTest: () => Promise<OcpConnectionTestResult | null>;
  refreshLeaseMetadata: () => Promise<OcpConnectionTestResult | null>;
  refreshLeaseStatus: () => Promise<OcpLeaseSchedulerStatus | null>;
  clearConnection: () => Promise<void>;
}

function isMissingConnectionError(error: unknown) {
  return error instanceof Error && error.message.includes("Connection profile not found");
}

function normalizeRequest(request: OcpConnectionRequest): OcpConnectionRequest {
  return {
    workspaceId: request.workspaceId?.trim() ?? "",
    clusterUrl: request.clusterUrl.trim(),
    authMode: request.authMode,
    verifySsl: request.verifySsl,
    defaultNamespace: request.defaultNamespace?.trim() ?? "",
    displayName: request.displayName?.trim() ?? "",
    saveProfile: request.saveProfile ?? false,
    token: request.token ?? "",
    username: request.username?.trim() ?? "",
    password: request.password ?? "",
  };
}

function buildSavedProfileId(request: OcpConnectionRequest) {
  const normalized = normalizeRequest(request);
  return [
    normalized.workspaceId?.toLowerCase() ?? "",
    normalized.clusterUrl.toLowerCase(),
    normalized.authMode,
    normalized.displayName?.toLowerCase() ?? "",
    normalized.defaultNamespace?.toLowerCase() ?? "",
  ].join("::");
}

function createRequestFromProfile(profile: OcpConnectionProfile): OcpConnectionRequest {
  return {
    workspaceId: profile.workspaceId,
    clusterUrl: profile.clusterUrl,
    authMode: profile.authMode,
    verifySsl: profile.verifySsl,
    defaultNamespace: profile.defaultNamespace,
    displayName: profile.displayName,
    saveProfile: profile.saveProfile,
    token: "",
    username: profile.usernameHint,
    password: "",
  };
}

function loadSavedProfiles(): SavedOcpConnectionProfile[] {
  if (typeof window === "undefined") return [];
  try {
    const raw = window.localStorage.getItem(SAVED_PROFILES_STORAGE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? (parsed as SavedOcpConnectionProfile[]) : [];
  } catch {
    return [];
  }
}

function findSavedProfileIdByConnection(
  savedProfiles: SavedOcpConnectionProfile[],
  profile: OcpConnectionProfile | null,
) {
  if (!profile) return null;
  const fallbackId = buildSavedProfileId(createRequestFromProfile(profile));
  const matched = savedProfiles.find((item) => item.id === fallbackId);
  return matched?.id ?? null;
}

export function useOcpConnection(workspaceId = ""): OcpConnectionController {
  const [submitState, setSubmitState] = useState<SubmitState>("idle");
  const [message, setMessage] = useState("클러스터 URL과 인증 정보를 입력한 뒤 연결과 검증을 진행합니다.");
  const [profile, setProfile] = useState<OcpConnectionProfile | null>(null);
  const [testResult, setTestResult] = useState<OcpConnectionTestResult | null>(null);
  const [schedulerStatus, setSchedulerStatus] = useState<OcpLeaseSchedulerStatus | null>(null);
  const [savedProfiles, setSavedProfiles] = useState<SavedOcpConnectionProfile[]>(() => loadSavedProfiles());
  const [activeSavedProfileId, setActiveSavedProfileId] = useState<string | null>(null);
  const visibleSavedProfiles = savedProfiles.filter(
    (item) => (item.workspaceId || "") === (workspaceId || ""),
  );

  useEffect(() => {
    if (!workspaceId) return;
    if (profile && (profile.workspaceId || "") !== workspaceId) {
      clearLocalConnection("선택된 workspace에 맞는 연결 프로필을 새로 선택하거나 생성하세요.");
    }
  }, [workspaceId, profile]);

  function clearLocalConnection(nextMessage = "저장된 세션 연결이 만료되었습니다. 다시 연결해야 합니다.") {
    setProfile(null);
    setTestResult(null);
    setActiveSavedProfileId(null);
    setMessage(nextMessage);
    window.sessionStorage.removeItem(PROFILE_STORAGE_KEY);
    window.sessionStorage.removeItem(TEST_RESULT_STORAGE_KEY);
    window.sessionStorage.removeItem(ACTIVE_SAVED_PROFILE_STORAGE_KEY);
  }

  function upsertSavedProfile(
    request: OcpConnectionRequest,
    nextProfile: OcpConnectionProfile,
    nextTestResult: OcpConnectionTestResult | null = null,
  ) {
    const normalizedRequest = normalizeRequest({
      ...request,
      displayName: request.displayName?.trim() || nextProfile.displayName,
      defaultNamespace: request.defaultNamespace?.trim() || nextProfile.defaultNamespace,
      username: request.username?.trim() || nextProfile.usernameHint,
    });
    const savedProfileId = buildSavedProfileId(normalizedRequest);

    setSavedProfiles((current) => {
      const existing = current.find((item) => item.id === savedProfileId);
      const nextSavedProfile: SavedOcpConnectionProfile = {
        id: savedProfileId,
        workspaceId: normalizedRequest.workspaceId ?? "",
        displayName: normalizedRequest.displayName || existing?.displayName || nextProfile.displayName || normalizedRequest.clusterUrl,
        clusterUrl: nextProfile.clusterUrl,
        authMode: nextProfile.authMode,
        verifySsl: nextProfile.verifySsl,
        defaultNamespace: normalizedRequest.defaultNamespace || existing?.defaultNamespace || nextProfile.defaultNamespace,
        usernameHint: nextProfile.usernameHint || normalizedRequest.username || existing?.usernameHint || "",
        lastResolvedUser:
          nextTestResult?.resolvedUser ||
          existing?.lastResolvedUser ||
          nextProfile.usernameHint ||
          normalizedRequest.displayName ||
          "",
        lastConnectionId: nextProfile.connectionId,
        lastStatus: nextProfile.status,
        lastConnectedAt: new Date().toISOString(),
        lastVerifiedAt:
          nextTestResult?.success
            ? new Date().toISOString()
            : existing?.lastVerifiedAt || nextProfile.lastVerifiedAt || "",
        request: normalizedRequest,
      };

      return [
        nextSavedProfile,
        ...current.filter((item) => item.id !== savedProfileId),
      ];
    });

    return savedProfileId;
  }

  function updateSavedProfileVerification(nextProfile: OcpConnectionProfile, result: OcpConnectionTestResult) {
    const fallbackId = buildSavedProfileId(createRequestFromProfile(nextProfile));
    const targetId = activeSavedProfileId ?? fallbackId;

    setSavedProfiles((current) =>
      current.map((item) => {
        if (item.id !== targetId) return item;
        return {
          ...item,
          clusterUrl: nextProfile.clusterUrl,
          defaultNamespace: result.resolvedNamespace || item.defaultNamespace || nextProfile.defaultNamespace,
          usernameHint: nextProfile.usernameHint || item.usernameHint,
          lastResolvedUser: result.resolvedUser || item.lastResolvedUser,
          lastConnectionId: nextProfile.connectionId,
          lastStatus: result.success ? "verified" : nextProfile.status,
          lastVerifiedAt: new Date().toISOString(),
          request: {
            ...item.request,
            defaultNamespace: result.resolvedNamespace || item.request.defaultNamespace || nextProfile.defaultNamespace,
          },
        };
      }),
    );

    if (!activeSavedProfileId) {
      setActiveSavedProfileId(targetId);
    }
  }

  async function refreshLeaseStatus() {
    try {
      const status = await getOcpLeaseSchedulerStatus();
      setSchedulerStatus(status);
      return status;
    } catch {
      return null;
    }
  }

  async function connectWithRequest(request: OcpConnectionRequest, progressMessage: string) {
    setSubmitState("connecting");
    setMessage(progressMessage);
    setTestResult(null);
    try {
      const status = await connectOcp(request);
      if (!status.connection) {
        setMessage(status.message || "연결 프로필 생성에 실패했습니다.");
        return null;
      }
      const savedProfileId = upsertSavedProfile(request, status.connection);
      setProfile(status.connection);
      setActiveSavedProfileId(savedProfileId);
      setMessage(status.message || "연결 프로필이 준비되었습니다.");
      await refreshLeaseStatus();
      return status.connection;
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "연결 프로필 생성 중 오류가 발생했습니다.");
      return null;
    } finally {
      setSubmitState("idle");
    }
  }

  useEffect(() => {
    void refreshLeaseStatus();

    const storedProfile = window.sessionStorage.getItem(PROFILE_STORAGE_KEY);
    const storedTestResult = window.sessionStorage.getItem(TEST_RESULT_STORAGE_KEY);
    const storedActiveSavedProfileId = window.sessionStorage.getItem(ACTIVE_SAVED_PROFILE_STORAGE_KEY);

    if (!storedProfile) {
      return;
    }

    try {
      const parsedProfile = JSON.parse(storedProfile) as OcpConnectionProfile;
      if (workspaceId && (parsedProfile.workspaceId || "") !== workspaceId) {
        clearLocalConnection("현재 workspace와 저장된 연결 세션이 달라서 연결 상태를 초기화했습니다.");
        return;
      }
      setProfile(parsedProfile);
      setActiveSavedProfileId(storedActiveSavedProfileId);
      if (storedTestResult) {
        setTestResult(JSON.parse(storedTestResult) as OcpConnectionTestResult);
      }

      void (async () => {
        try {
          const status = await getOcpConnectionStatus(parsedProfile.connectionId);
          if (!status.connected || !status.connection) {
            clearLocalConnection("백엔드가 재시작되어 기존 연결 세션이 사라졌습니다. 다시 연결해야 합니다.");
            return;
          }
          setProfile(status.connection);
          setMessage(status.message || "저장된 연결 세션을 불러왔습니다.");
          setActiveSavedProfileId((current) => current ?? findSavedProfileIdByConnection(savedProfiles, status.connection));
        } catch {
          clearLocalConnection("저장된 연결 상태를 복원하지 못했습니다. 다시 연결해야 합니다.");
        }
      })();
    } catch {
      clearLocalConnection();
    }
  }, [workspaceId]);

  useEffect(() => {
    window.localStorage.setItem(SAVED_PROFILES_STORAGE_KEY, JSON.stringify(savedProfiles));
  }, [savedProfiles]);

  useEffect(() => {
    if (profile) {
      window.sessionStorage.setItem(PROFILE_STORAGE_KEY, JSON.stringify(profile));
    } else {
      window.sessionStorage.removeItem(PROFILE_STORAGE_KEY);
    }
  }, [profile]);

  useEffect(() => {
    if (testResult) {
      window.sessionStorage.setItem(TEST_RESULT_STORAGE_KEY, JSON.stringify(testResult));
    } else {
      window.sessionStorage.removeItem(TEST_RESULT_STORAGE_KEY);
    }
  }, [testResult]);

  useEffect(() => {
    if (activeSavedProfileId) {
      window.sessionStorage.setItem(ACTIVE_SAVED_PROFILE_STORAGE_KEY, activeSavedProfileId);
    } else {
      window.sessionStorage.removeItem(ACTIVE_SAVED_PROFILE_STORAGE_KEY);
    }
  }, [activeSavedProfileId]);

  async function createConnection(request: OcpConnectionRequest) {
    return connectWithRequest({ ...request, workspaceId }, "연결 프로필을 생성하는 중입니다.");
  }

  async function connectSavedProfile(savedProfileId: string) {
    const savedProfile = visibleSavedProfiles.find((item) => item.id === savedProfileId);
    if (!savedProfile) {
      setMessage("선택한 저장 프로필을 찾을 수 없습니다.");
      return null;
    }
    return connectWithRequest(savedProfile.request, `${savedProfile.displayName || savedProfile.clusterUrl} 프로필로 다시 연결하는 중입니다.`);
  }

  function removeSavedProfile(savedProfileId: string) {
    const removedProfile = visibleSavedProfiles.find((item) => item.id === savedProfileId);
    setSavedProfiles((current) => current.filter((item) => item.id !== savedProfileId));
    setActiveSavedProfileId((current) => (current === savedProfileId ? null : current));
    setMessage(`${removedProfile?.displayName || removedProfile?.clusterUrl || "선택한"} 저장 프로필을 삭제했습니다.`);
  }

  async function runConnectionTest() {
    if (!profile) return null;
    setSubmitState("testing");
    setMessage("클러스터 연결을 검증하는 중입니다.");
    try {
      const result = await testOcpConnection(profile.connectionId);
      setTestResult(result);
      updateSavedProfileVerification(profile, result);
      await refreshLeaseStatus();
      setMessage(result.message || (result.success ? "연결 검증이 완료되었습니다." : "연결 검증에 실패했습니다."));
      return result;
    } catch (error) {
      if (isMissingConnectionError(error)) {
        clearLocalConnection("백엔드에 연결 프로필이 없어졌습니다. 다시 연결해야 합니다.");
        return null;
      }
      setMessage(error instanceof Error ? error.message : "연결 테스트 중 오류가 발생했습니다.");
      return null;
    } finally {
      setSubmitState("idle");
    }
  }

  async function refreshLeaseMetadata() {
    if (!profile) return null;
    setSubmitState("testing");
    setMessage("비밀값 lease 메타데이터를 갱신하는 중입니다.");
    try {
      const result = await refreshOcpConnectionLease(profile.connectionId);
      setTestResult(result);
      updateSavedProfileVerification(profile, result);
      await refreshLeaseStatus();
      setMessage(result.message || "비밀값 lease 메타데이터를 갱신했습니다.");
      return result;
    } catch (error) {
      if (isMissingConnectionError(error)) {
        clearLocalConnection("백엔드에 연결 프로필이 없어졌습니다. 다시 연결해야 합니다.");
        return null;
      }
      setMessage(error instanceof Error ? error.message : "lease 메타데이터 갱신 중 오류가 발생했습니다.");
      return null;
    } finally {
      setSubmitState("idle");
    }
  }

  async function clearConnection() {
    if (!profile) return;
    setSubmitState("disconnecting");
    try {
      await disconnectOcp(profile.connectionId);
      setProfile(null);
      setTestResult(null);
      setActiveSavedProfileId(null);
      setMessage("현재 연결을 로그아웃했습니다. 저장된 프로필로 언제든 다시 연결할 수 있습니다.");
      await refreshLeaseStatus();
    } catch (error) {
      if (isMissingConnectionError(error)) {
        clearLocalConnection("백엔드에 연결 프로필이 이미 존재하지 않습니다.");
      } else {
        setMessage(error instanceof Error ? error.message : "연결 해제 중 오류가 발생했습니다.");
      }
    } finally {
      setSubmitState("idle");
    }
  }

  return {
    submitState,
    isBusy: submitState !== "idle",
    message,
    profile,
    testResult,
    schedulerStatus,
    savedProfiles: visibleSavedProfiles,
    activeSavedProfileId,
    createConnection,
    connectSavedProfile,
    removeSavedProfile,
    runConnectionTest,
    refreshLeaseMetadata,
    refreshLeaseStatus,
    clearConnection,
  };
}

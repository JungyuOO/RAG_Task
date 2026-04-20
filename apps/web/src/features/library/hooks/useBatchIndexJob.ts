import { useEffect, useRef, useState } from "react";

import type { BatchIndexRequest, BatchJobStatusResponse } from "@/features/library/types";
import { cancelBatchIndexJob, getBatchIndexJob, listBatchIndexJobs, retryFailedBatchIndexJob, submitBatchIndexJob } from "@/features/library/api/indexingApi";

export function useBatchIndexJob() {
  const [job, setJob] = useState<BatchJobStatusResponse | null>(null);
  const [history, setHistory] = useState<BatchJobStatusResponse[]>([]);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState("");
  const pollTimer = useRef<number | null>(null);

  useEffect(() => {
    void refreshHistory();
    return () => {
      if (pollTimer.current !== null) {
        window.clearTimeout(pollTimer.current);
      }
    };
  }, []);

  function schedulePoll(jobId: string) {
    if (pollTimer.current !== null) {
      window.clearTimeout(pollTimer.current);
    }
    pollTimer.current = window.setTimeout(async () => {
      try {
        const next = await getBatchIndexJob(jobId);
        setJob(next);
        await refreshHistory();
        if (next.status === "pending" || next.status === "running") {
          schedulePoll(jobId);
        }
      } catch (nextError) {
        setError(nextError instanceof Error ? nextError.message : "batch job 상태를 불러오지 못했습니다.");
      }
    }, 1200);
  }

  async function submit(request: BatchIndexRequest) {
    setIsSubmitting(true);
    setError("");
    try {
      const next = await submitBatchIndexJob(request);
      setJob(next);
      await refreshHistory();
      if (next.status === "pending" || next.status === "running") {
        schedulePoll(next.jobId);
      }
    } catch (nextError) {
      setError(nextError instanceof Error ? nextError.message : "batch job 제출에 실패했습니다.");
    } finally {
      setIsSubmitting(false);
    }
  }

  async function retryFailed(jobId: string) {
    setIsSubmitting(true);
    setError("");
    try {
      const next = await retryFailedBatchIndexJob(jobId);
      setJob(next);
      await refreshHistory();
      if (next.status === "pending" || next.status === "running") {
        schedulePoll(next.jobId);
      }
    } catch (nextError) {
      setError(nextError instanceof Error ? nextError.message : "실패 항목 재시도에 실패했습니다.");
    } finally {
      setIsSubmitting(false);
    }
  }

  async function cancel(jobId: string) {
    setIsSubmitting(true);
    setError("");
    try {
      const next = await cancelBatchIndexJob(jobId);
      setJob(next);
      await refreshHistory();
    } catch (nextError) {
      setError(nextError instanceof Error ? nextError.message : "batch job 취소에 실패했습니다.");
    } finally {
      setIsSubmitting(false);
    }
  }

  async function refreshHistory() {
    try {
      const jobs = await listBatchIndexJobs(10);
      setHistory(jobs);
    } catch {
      // non-fatal history refresh
    }
  }

  return {
    job,
    history,
    isSubmitting,
    error,
    submit,
    retryFailed,
    cancel,
    refreshHistory,
  };
}




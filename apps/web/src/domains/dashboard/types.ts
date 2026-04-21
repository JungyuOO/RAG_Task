export type WorkspaceRecommendationRecord = {
  recommendationId: string;
  workspaceId: string;
  connectionId: string;
  namespace: string;
  resourceKind: string;
  resourceName: string;
  recommendationType: string;
  riskLevel: string;
  summary: string;
  rationale: string;
  createdAt: string;
};

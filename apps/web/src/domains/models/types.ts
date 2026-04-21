export type WorkspaceModelProfile = {
  workspaceId: string;
  chatProvider: string;
  chatBaseUrl: string;
  chatModel: string;
  chatApiKeyMode: string;
  embeddingProvider: string;
  embeddingBaseUrl: string;
  embeddingModel: string;
  embeddingApiKeyMode: string;
  updatedAt: string;
};

export type WorkspaceModelProfileUpdateRequest = {
  chatProvider: string;
  chatBaseUrl: string;
  chatModel: string;
  chatApiKeyMode: string;
  embeddingProvider: string;
  embeddingBaseUrl: string;
  embeddingModel: string;
  embeddingApiKeyMode: string;
};

export type WorkspaceRecord = {
  workspaceId: string;
  name: string;
  slug: string;
  industry: string;
  environment: string;
  createdAt: string;
  updatedAt: string;
};

export type WorkspaceCreateRequest = {
  name: string;
  environment?: string;
};

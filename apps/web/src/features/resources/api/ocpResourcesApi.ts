import type {
  OcpDashboardMetricsResponse,
  OcpLiveResourceDetailResponse,
  OcpLiveNamespaceListResponse,
  OcpLiveResourceListResponse,
  OcpLiveResourceSummary,
  OcpMetricSeries,
  OcpOverviewResponse,
} from "@/features/connection/types";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "";

function apiUrl(path: string) {
  return `${API_BASE_URL}${path}`;
}

type ApiOcpLiveResourceSummary = {
  name: string;
  namespace: string;
  kind: string;
  created_at: string;
  phase: string;
  node_name: string;
  ready_replicas: number;
  replicas: number;
  type: string;
  cluster_ip: string;
  host: string;
  to: string;
};

type ApiOcpNamespaceListResponse = {
  connection_id: string;
  cluster_url: string;
  count: number;
  items: string[];
};

type ApiOcpResourceListResponse = {
  connection_id: string;
  cluster_url: string;
  resource: string;
  namespace: string;
  count: number;
  items: ApiOcpLiveResourceSummary[];
};

type ApiOcpResourceDetailResponse = {
  connection_id: string;
  cluster_url: string;
  resource: string;
  namespace: string;
  name: string;
  kind: string;
  manifest_yaml: string;
  manifest_json: Record<string, unknown>;
};

type ApiOcpOverviewResponse = {
  connection_id: string;
  cluster_url: string;
  default_namespace: string;
  namespace_count: number;
  namespace_sample: string[];
  resource_counts: Record<string, number>;
  message: string;
};

type ApiOcpMetricPoint = {
  timestamp: number;
  value: number;
};

type ApiOcpMetricSeries = {
  metric_id: string;
  label: string;
  unit: string;
  current_value: number;
  capacity_value: number;
  available_value: number;
  points: ApiOcpMetricPoint[];
};

type ApiOcpDashboardMetricsResponse = {
  connection_id: string;
  cluster_url: string;
  window: string;
  step: string;
  series: ApiOcpMetricSeries[];
};

async function readJson<T>(response: Response): Promise<T> {
  const text = await response.text();
  if (!response.ok) {
    throw new Error(text || `HTTP ${response.status}`);
  }
  return JSON.parse(text) as T;
}

function mapResourceSummary(input: ApiOcpLiveResourceSummary): OcpLiveResourceSummary {
  return {
    name: input.name,
    namespace: input.namespace,
    kind: input.kind,
    createdAt: input.created_at,
    phase: input.phase,
    nodeName: input.node_name,
    readyReplicas: input.ready_replicas,
    replicas: input.replicas,
    type: input.type,
    clusterIp: input.cluster_ip,
    host: input.host,
    to: input.to,
  };
}

function mapNamespaces(input: ApiOcpNamespaceListResponse): OcpLiveNamespaceListResponse {
  return {
    connectionId: input.connection_id,
    clusterUrl: input.cluster_url,
    count: input.count,
    items: input.items,
  };
}

function mapResources(input: ApiOcpResourceListResponse): OcpLiveResourceListResponse {
  return {
    connectionId: input.connection_id,
    clusterUrl: input.cluster_url,
    resource: input.resource,
    namespace: input.namespace,
    count: input.count,
    items: input.items.map(mapResourceSummary),
  };
}

function mapResourceDetail(input: ApiOcpResourceDetailResponse): OcpLiveResourceDetailResponse {
  return {
    connectionId: input.connection_id,
    clusterUrl: input.cluster_url,
    resource: input.resource,
    namespace: input.namespace,
    name: input.name,
    kind: input.kind,
    manifestYaml: input.manifest_yaml,
    manifestJson: input.manifest_json ?? {},
  };
}

function mapOverview(input: ApiOcpOverviewResponse): OcpOverviewResponse {
  return {
    connectionId: input.connection_id,
    clusterUrl: input.cluster_url,
    defaultNamespace: input.default_namespace,
    namespaceCount: input.namespace_count,
    namespaceSample: input.namespace_sample,
    resourceCounts: input.resource_counts,
    message: input.message,
  };
}

function mapMetricSeries(input: ApiOcpMetricSeries): OcpMetricSeries {
  return {
    metricId: input.metric_id,
    label: input.label,
    unit: input.unit,
    currentValue: input.current_value,
    capacityValue: input.capacity_value,
    availableValue: input.available_value,
    points: (input.points ?? []).map((point) => ({
      timestamp: point.timestamp,
      value: point.value,
    })),
  };
}

function mapDashboardMetrics(input: ApiOcpDashboardMetricsResponse): OcpDashboardMetricsResponse {
  return {
    connectionId: input.connection_id,
    clusterUrl: input.cluster_url,
    window: input.window,
    step: input.step,
    series: (input.series ?? []).map(mapMetricSeries),
  };
}

export async function getOcpOverview(connectionId: string): Promise<OcpOverviewResponse> {
  const response = await fetch(apiUrl(`/api/v1/ocp/overview/${connectionId}`));
  return mapOverview(await readJson<ApiOcpOverviewResponse>(response));
}

export async function getOcpDashboardMetrics(
  connectionId: string,
  window = "1h",
  step = "5m",
): Promise<OcpDashboardMetricsResponse> {
  const search = new URLSearchParams({ window, step });
  const response = await fetch(apiUrl(`/api/v1/ocp/metrics/${connectionId}?${search.toString()}`));
  return mapDashboardMetrics(await readJson<ApiOcpDashboardMetricsResponse>(response));
}

export async function getOcpNamespaces(connectionId: string): Promise<OcpLiveNamespaceListResponse> {
  const response = await fetch(apiUrl(`/api/v1/ocp/namespaces/${connectionId}`));
  return mapNamespaces(await readJson<ApiOcpNamespaceListResponse>(response));
}

export async function getOcpResources(
  connectionId: string,
  resource: "pods" | "deployments" | "services" | "routes" | "events",
  namespace: string,
): Promise<OcpLiveResourceListResponse> {
  const search = new URLSearchParams({ resource, namespace });
  const response = await fetch(apiUrl(`/api/v1/ocp/resources/${connectionId}?${search.toString()}`));
  return mapResources(await readJson<ApiOcpResourceListResponse>(response));
}

export async function getOcpResourceDetail(
  connectionId: string,
  resource: "pods" | "deployments" | "services" | "routes" | "events",
  namespace: string,
  name: string,
): Promise<OcpLiveResourceDetailResponse> {
  const search = new URLSearchParams({ resource, namespace, name });
  const response = await fetch(apiUrl(`/api/v1/ocp/resource-detail/${connectionId}?${search.toString()}`));
  return mapResourceDetail(await readJson<ApiOcpResourceDetailResponse>(response));
}



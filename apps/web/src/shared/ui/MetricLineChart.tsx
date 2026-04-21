import type { OcpMetricPoint } from "@/domains/connection/types";

type MetricLineChartProps = {
  points: OcpMetricPoint[];
  capacity?: number;
};

export function MetricLineChart({ points, capacity = 0 }: MetricLineChartProps) {
  if (!points.length) {
    return <div className="metric-line-chart metric-line-chart--empty">No data</div>;
  }

  const width = 360;
  const height = 88;
  const paddingX = 8;
  const paddingY = 8;
  const values = points.map((point) => point.value);
  const maxValue = Math.max(capacity || 0, ...values, 1);
  const minValue = 0;
  const span = Math.max(maxValue - minValue, 1);

  const coordinates = points.map((point, index) => {
    const x = paddingX + (index / Math.max(points.length - 1, 1)) * (width - paddingX * 2);
    const y = height - paddingY - ((point.value - minValue) / span) * (height - paddingY * 2);
    return `${x},${y}`;
  });

  const areaPoints = [`${paddingX},${height - paddingY}`, ...coordinates, `${width - paddingX},${height - paddingY}`].join(" ");
  const linePoints = coordinates.join(" ");
  const capacityY = capacity > 0 ? height - paddingY - ((capacity - minValue) / span) * (height - paddingY * 2) : null;

  return (
    <svg viewBox={`0 0 ${width} ${height}`} className="metric-line-chart" role="img" aria-label="metric trend">
      {capacityY !== null ? (
        <line
          x1={paddingX}
          x2={width - paddingX}
          y1={capacityY}
          y2={capacityY}
          className="metric-line-chart__capacity"
        />
      ) : null}
      <polygon points={areaPoints} className="metric-line-chart__area" />
      <polyline points={linePoints} className="metric-line-chart__line" />
    </svg>
  );
}




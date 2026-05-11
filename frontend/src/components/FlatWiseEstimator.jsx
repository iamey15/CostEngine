import { AlertTriangle, ArrowDown, ArrowLeft, ArrowRight, ArrowUp, BadgeCheck, BrainCircuit, CheckCircle2, FileUp, Home, Layers, Loader2, Maximize2, MousePointer2, Plus, RefreshCw, SlidersHorizontal, Sparkles, Trash2, UploadCloud, Wand2, X } from "lucide-react";
import { useMemo, useRef, useState } from "react";
import { Bar, BarChart, Cell, Pie, PieChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { endpoints } from "../api";
import { money, shortMoney } from "../utils";
import Button from "./Button";

const COLORS = ["#4F46E5", "#0891B2", "#7C3AED", "#16A34A", "#D97706"];

const editableFields = [
  ["room_count", "Primary rooms"],
  ["bedroom_count", "Bedrooms"],
  ["bathroom_count", "Bathrooms"],
  ["kitchen_count", "Kitchens"],
  ["hall_count", "Hall / Living"],
  ["wall_thickness_ft", "Wall thickness ft"],
  ["carpet_area_sqft", "Carpet area"],
  ["built_up_area_sqft", "Built-up area"],
  ["external_wall_length_ft", "External wall length"],
  ["internal_wall_length_ft", "Internal wall length"],
  ["wall_height_ft", "Wall height"],
];

const zoneTypes = [
  ["bedroom", "Bedroom"],
  ["bathroom", "Bathroom / Toilet"],
  ["kitchen", "Kitchen"],
  ["living", "Living / Hall"],
  ["balcony", "Balcony / Veranda"],
  ["service", "Service / Stair / Utility"],
  ["storage", "Storage"],
  ["unlabeled", "Unlabeled"],
];

const polygonFromRoom = (room) => {
  if (Array.isArray(room.polygon) && room.polygon.length >= 3) {
    return room.polygon.map((point) => [Number(point[0]), Number(point[1])]);
  }
  const x = Number(room.x || 8);
  const y = Number(room.y || 8);
  const width = Number(room.width || 18);
  const height = Number(room.height || 14);
  return [
    [x, y],
    [x + width, y],
    [x + width, y + height],
    [x, y + height],
  ];
};

const bboxFromPolygon = (polygon) => {
  const xs = polygon.map((point) => point[0]);
  const ys = polygon.map((point) => point[1]);
  return {
    minX: Math.min(...xs),
    maxX: Math.max(...xs),
    minY: Math.min(...ys),
    maxY: Math.max(...ys),
    width: Math.max(0.01, Math.max(...xs) - Math.min(...xs)),
    height: Math.max(0.01, Math.max(...ys) - Math.min(...ys)),
  };
};

const scalePolygonToBox = (polygon, nextBox) => {
  const box = bboxFromPolygon(polygon);
  const width = Math.max(4, nextBox.maxX - nextBox.minX);
  const height = Math.max(4, nextBox.maxY - nextBox.minY);
  return polygon.map(([x, y]) => [
    nextBox.minX + ((x - box.minX) / box.width) * width,
    nextBox.minY + ((y - box.minY) / box.height) * height,
  ]);
};

const roomFromPolygon = (room, polygon) => {
  const clamped = polygon.map(([x, y]) => [Math.max(0, Math.min(125, x)), Math.max(0, Math.min(92, y))]);
  const xs = clamped.map((point) => point[0]);
  const ys = clamped.map((point) => point[1]);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  let polygonArea = 0;
  clamped.forEach((point, index) => {
    const next = clamped[(index + 1) % clamped.length];
    polygonArea += point[0] * next[1] - next[0] * point[1];
  });
  const areaSqft = Math.max(20, (Math.abs(polygonArea) / 2 / (125 * 92)) * 1250 * 1.08);
  return {
    ...room,
    polygon: clamped.map(([x, y]) => [Math.round(x * 10) / 10, Math.round(y * 10) / 10]),
    x: Math.round(minX * 10) / 10,
    y: Math.round(minY * 10) / 10,
    width: Math.round(Math.max(4, maxX - minX) * 10) / 10,
    height: Math.round(Math.max(4, maxY - minY) * 10) / 10,
    area_sqft: Math.round(areaSqft * 10) / 10,
  };
};

const labelPositionFromRoom = (room) => {
  const polygon = polygonFromRoom(room);
  const box = bboxFromPolygon(polygon);
  const baseX = Number.isFinite(Number(room.label_x))
    ? Number(room.label_x)
    : Math.min(...polygon.map((point) => point[0])) + 2;
  const baseY = Number.isFinite(Number(room.label_y))
    ? Number(room.label_y)
    : Math.min(...polygon.map((point) => point[1])) + 6.5;
  return {
    x: Math.max(box.minX + 1.2, Math.min(box.maxX - 10, baseX)),
    y: Math.max(box.minY + 4.8, Math.min(box.maxY - 4.5, baseY)),
    areaY: Number.isFinite(Number(room.label_area_y))
      ? Number(room.label_area_y)
      : Math.max(box.minY + 8.8, Math.min(box.maxY - 1.4, baseY + 4.5)),
  };
};

function DetectionPreview({ detection, previewUrl, imageSize }) {
  const rooms = detection?.rooms || [];
  const hasPreview = Boolean(previewUrl);
  const showUploadPanel = hasPreview || !rooms.length;
  const aspectStyle = Array.isArray(imageSize) && imageSize[0] && imageSize[1]
    ? { aspectRatio: `${imageSize[0]} / ${imageSize[1]}` }
    : { aspectRatio: "4 / 3" };
  return (
    <div className={`grid gap-3 ${showUploadPanel ? "lg:grid-cols-2" : "lg:grid-cols-1"}`}>
      {showUploadPanel ? (
        <div className="overflow-hidden rounded-2xl border border-slate-200 bg-white shadow-sm">
          <div className="flex items-center justify-between border-b border-slate-200 bg-slate-50 px-4 py-3">
            <p className="text-sm font-bold text-[#111827]">Uploaded floorplan</p>
            <span className="rounded-md bg-white px-2 py-1 text-xs font-bold text-[#64748B]">Original</span>
          </div>
          <div className="flex min-h-80 items-center justify-center bg-white p-3">
            {previewUrl ? (
              <img src={previewUrl} alt="Uploaded floorplan preview" className="max-h-[34rem] w-full object-contain" />
            ) : (
              <div className="rounded-xl bg-slate-50 px-4 py-3 text-sm font-semibold text-[#64748B]">Preview appears here after image upload</div>
            )}
          </div>
        </div>
      ) : null}

      <div className="overflow-hidden rounded-2xl border border-slate-200 bg-white shadow-sm">
        <div className="flex items-center justify-between border-b border-slate-200 bg-slate-50 px-4 py-3">
          <p className="text-sm font-bold text-[#111827]">Detected layout</p>
          <span className="rounded-md bg-[#EEF2FF] px-2 py-1 text-xs font-bold text-[#3730A3]">{rooms.length ? `${rooms.length} raw zones` : "Waiting"}</span>
        </div>
        <div className="relative overflow-hidden bg-[#F8FAFC]" style={aspectStyle}>
          <svg viewBox="0 0 125 92" className="absolute inset-0 h-full w-full">
            {previewUrl ? (
              <image href={previewUrl} x="0" y="0" width="125" height="92" preserveAspectRatio="none" opacity="0.42" />
            ) : (
              <rect x="3" y="3" width="116" height="84" rx="2" fill="rgba(255,255,255,0.82)" stroke="#334155" strokeWidth="1.2" />
            )}
            {rooms.map((room, index) => {
              const polygon = polygonFromRoom(room);
              const points = polygon.map(([x, y]) => `${x},${y}`).join(" ");
              const label = labelPositionFromRoom(room);
              return (
                <g key={room.id}>
                  <polygon
                    points={points}
                    fill={COLORS[index % COLORS.length]}
                    fillOpacity="0.18"
                    stroke={COLORS[index % COLORS.length]}
                    strokeWidth="1.15"
                    strokeDasharray={room.confidence < 0.78 ? "2 2" : "0"}
                  />
                  <text x={label.x} y={label.y} fill="#111827" fontSize="3.4" fontWeight="700">
                    {room.label}
                  </text>
                  <text x={label.x} y={label.areaY} fill="#475569" fontSize="2.8">
                    {Math.round(room.area_sqft)} sqft
                  </text>
                </g>
              );
            })}
          </svg>
          <div className="absolute bottom-3 left-3 rounded-xl border border-slate-200 bg-white/90 px-3 py-2 text-xs font-semibold text-[#475569] shadow-sm backdrop-blur">
            AI overlay: raw zones, labels, wall continuity, area cross-check
          </div>
        </div>
      </div>
    </div>
  );
}

function ZoneEditor({ rooms, setRooms, previewUrl, selectedId, setSelectedId, onAutoLabel, loading, onClose }) {
  const svgRef = useRef(null);
  const dragRef = useRef(null);
  const [selectedVertex, setSelectedVertex] = useState(null);
  const selectedRoom = rooms.find((room) => room.id === selectedId) || rooms[0];

  const pointFromEvent = (event) => {
    const rect = svgRef.current.getBoundingClientRect();
    return {
      x: Math.max(0, Math.min(125, ((event.clientX - rect.left) / rect.width) * 125)),
      y: Math.max(0, Math.min(92, ((event.clientY - rect.top) / rect.height) * 92)),
    };
  };

  const updateRoom = (id, patcher) => {
    setRooms((current) =>
      current.map((room) => {
        if (room.id !== id) return room;
        const patchIsFunction = typeof patcher === "function";
        const patchKeys = patchIsFunction ? [] : Object.keys(patcher);
        const next = patchIsFunction ? patcher(room) : { ...room, ...patcher };
        let polygon = polygonFromRoom(next);
        if (!patchIsFunction && patchKeys.some((key) => ["x", "y", "width", "height"].includes(key))) {
          const currentPolygon = polygonFromRoom(room);
          const currentBox = bboxFromPolygon(currentPolygon);
          const x = Math.max(0, Math.min(Number(next.x ?? currentBox.minX), 121));
          const y = Math.max(0, Math.min(Number(next.y ?? currentBox.minY), 88));
          const width = Math.max(4, Math.min(Number(next.width ?? currentBox.width), 125 - x));
          const height = Math.max(4, Math.min(Number(next.height ?? currentBox.height), 92 - y));
          polygon = scalePolygonToBox(currentPolygon, { minX: x, minY: y, maxX: x + width, maxY: y + height });
        }
        return {
          ...roomFromPolygon(next, polygon),
          border_thickness_ft: Math.round(Number(next.border_thickness_ft || 0.5) * 100) / 100,
        };
      })
    );
  };

  const moveLayer = (id, direction) => {
    setRooms((current) => {
      const index = current.findIndex((room) => room.id === id);
      const nextIndex = index + direction;
      if (index < 0 || nextIndex < 0 || nextIndex >= current.length) return current;
      const copy = [...current];
      const [item] = copy.splice(index, 1);
      copy.splice(nextIndex, 0, item);
      return copy;
    });
  };

  const onPointerDown = (event, room, mode = "move", vertexIndex = null) => {
    event.preventDefault();
    event.stopPropagation();
    setSelectedId(room.id);
    setSelectedVertex(vertexIndex);
    const point = pointFromEvent(event);
    dragRef.current = { id: room.id, mode, vertexIndex, start: point, room: { ...room, polygon: polygonFromRoom(room) } };
    event.currentTarget.setPointerCapture?.(event.pointerId);
  };

  const onEdgePointerDown = (event, room, edgeIndex) => {
    event.preventDefault();
    event.stopPropagation();
    const pointer = pointFromEvent(event);
    const polygon = polygonFromRoom(room);
    const start = polygon[edgeIndex];
    const end = polygon[(edgeIndex + 1) % polygon.length];
    const midpoint = [(start[0] + end[0]) / 2, (start[1] + end[1]) / 2];
    const inserted = [...polygon.slice(0, edgeIndex + 1), midpoint, ...polygon.slice(edgeIndex + 1)];
    const nextRoom = roomFromPolygon(room, inserted);
    setSelectedId(room.id);
    setSelectedVertex(edgeIndex + 1);
    updateRoom(room.id, () => nextRoom);
    dragRef.current = { id: room.id, mode: "vertex", vertexIndex: edgeIndex + 1, start: pointer, room: nextRoom };
    event.currentTarget.setPointerCapture?.(event.pointerId);
  };

  const onResizePointerDown = (event, room, handle) => {
    event.preventDefault();
    event.stopPropagation();
    setSelectedId(room.id);
    setSelectedVertex(null);
    const point = pointFromEvent(event);
    const polygon = polygonFromRoom(room);
    dragRef.current = { id: room.id, mode: "resize", handle, start: point, room: { ...room, polygon }, box: bboxFromPolygon(polygon) };
    event.currentTarget.setPointerCapture?.(event.pointerId);
  };

  const onPointerMove = (event) => {
    if (!dragRef.current) return;
    const point = pointFromEvent(event);
    const { id, mode, vertexIndex, start, room, handle, box } = dragRef.current;
    const dx = point.x - start.x;
    const dy = point.y - start.y;
    if (mode === "vertex") {
      const polygon = polygonFromRoom(room).map((vertex, index) => (index === vertexIndex ? [vertex[0] + dx, vertex[1] + dy] : vertex));
      updateRoom(id, (current) => roomFromPolygon(current, polygon));
    } else if (mode === "resize") {
      const nextBox = { ...box };
      if (handle.includes("w")) nextBox.minX = Math.min(box.maxX - 4, Math.max(0, box.minX + dx));
      if (handle.includes("e")) nextBox.maxX = Math.max(box.minX + 4, Math.min(125, box.maxX + dx));
      if (handle.includes("n")) nextBox.minY = Math.min(box.maxY - 4, Math.max(0, box.minY + dy));
      if (handle.includes("s")) nextBox.maxY = Math.max(box.minY + 4, Math.min(92, box.maxY + dy));
      const polygon = scalePolygonToBox(polygonFromRoom(room), nextBox);
      updateRoom(id, (current) => roomFromPolygon(current, polygon));
    } else {
      const polygon = polygonFromRoom(room).map(([x, y]) => [x + dx, y + dy]);
      updateRoom(id, (current) => roomFromPolygon(current, polygon));
    }
  };

  const stopDrag = () => {
    dragRef.current = null;
  };

  const addZone = () => {
    const id = `edited-zone-${Date.now()}`;
    setRooms((current) => [
      ...current,
      roomFromPolygon({
        id,
        type: "unlabeled",
        label: "Pending AI label",
        x: 18,
        y: 18,
        width: 22,
        height: 16,
        area_sqft: 120,
        confidence: 0.52,
        border_thickness_ft: 0.5,
      }, [[18, 18], [40, 18], [40, 34], [18, 34]]),
    ]);
    setSelectedId(id);
    setSelectedVertex(null);
  };

  const deleteZone = () => {
    if (!selectedId) return;
    setRooms((current) => current.filter((room) => room.id !== selectedId));
    setSelectedId(null);
    setSelectedVertex(null);
  };

  const addVertex = () => {
    if (!selectedRoom) return;
    const polygon = polygonFromRoom(selectedRoom);
    let longestIndex = 0;
    let longestLength = -1;
    polygon.forEach((point, index) => {
      const next = polygon[(index + 1) % polygon.length];
      const length = Math.hypot(next[0] - point[0], next[1] - point[1]);
      if (length > longestLength) {
        longestLength = length;
        longestIndex = index;
      }
    });
    const point = polygon[longestIndex];
    const next = polygon[(longestIndex + 1) % polygon.length];
    const midpoint = [(point[0] + next[0]) / 2, (point[1] + next[1]) / 2];
    const updated = [...polygon.slice(0, longestIndex + 1), midpoint, ...polygon.slice(longestIndex + 1)];
    updateRoom(selectedRoom.id, (room) => roomFromPolygon(room, updated));
    setSelectedVertex(longestIndex + 1);
  };

  const deleteVertex = () => {
    if (!selectedRoom || selectedVertex === null) return;
    const polygon = polygonFromRoom(selectedRoom);
    if (polygon.length <= 3) return;
    const updated = polygon.filter((_, index) => index !== selectedVertex);
    updateRoom(selectedRoom.id, (room) => roomFromPolygon(room, updated));
    setSelectedVertex(null);
  };

  return (
    <div className="fixed inset-0 z-[80] flex flex-col bg-[#F8FAFC] text-[#111827]">
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-200 bg-white px-5 py-3 shadow-sm">
        <div>
          <p className="text-xs font-bold uppercase tracking-wide text-[#4F46E5]">Full Screen Canvas</p>
          <h3 className="text-lg font-bold">Correct zones and labels</h3>
        </div>
        <div className="flex flex-wrap gap-2">
          <Button type="button" variant="secondary" icon={Plus} className="px-3 py-2" onClick={addZone}>Add Zone</Button>
          <Button type="button" variant="secondary" icon={Trash2} className="px-3 py-2" disabled={!selectedId} onClick={deleteZone}>Delete</Button>
          <Button type="button" icon={Wand2} loading={loading === "relabel"} disabled={!rooms.length} onClick={onAutoLabel}>AI Re-check Zones</Button>
          <Button type="button" variant="secondary" icon={X} className="px-3 py-2" onClick={onClose}>Close</Button>
        </div>
      </div>

      <div className="grid min-h-0 flex-1 grid-cols-[260px_minmax(0,1fr)_320px] overflow-hidden">
        <aside className="min-h-0 overflow-y-auto border-r border-slate-200 bg-white">
          <div className="sticky top-0 z-10 border-b border-slate-200 bg-white px-4 py-3">
            <div className="flex items-center gap-2">
              <Layers className="h-4 w-4 text-[#4F46E5]" />
              <p className="text-sm font-bold">Layers</p>
            </div>
            <p className="mt-1 text-xs text-[#64748B]">Top item draws above lower items.</p>
          </div>
          <div className="space-y-2 p-3">
            {[...rooms].reverse().map((room, reverseIndex) => {
              const index = rooms.length - 1 - reverseIndex;
              const selected = selectedId === room.id;
              return (
                <button
                  key={room.id}
                  type="button"
                  onClick={() => setSelectedId(room.id)}
                  className={`w-full rounded-xl border p-3 text-left transition ${selected ? "border-[#4F46E5] bg-[#EEF2FF] text-[#3730A3]" : "border-slate-200 bg-white hover:bg-slate-50"}`}
                >
                  <div className="flex items-center justify-between gap-2">
                    <span className="truncate text-sm font-bold">{room.label || `Zone ${index + 1}`}</span>
                    <span className="rounded-md bg-white/80 px-2 py-1 text-[10px] font-bold uppercase">{room.type || "zone"}</span>
                  </div>
                  <p className="mt-1 text-xs text-[#64748B]">{Math.round(room.area_sqft || 0)} sqft | {room.border_thickness_ft || 0.5} ft wall</p>
                </button>
              );
            })}
          </div>
        </aside>

        <main className="relative min-h-0 overflow-hidden bg-[#E2E8F0]">
          <div className="absolute inset-0 overflow-hidden bg-white">
            <svg ref={svgRef} viewBox="0 0 125 92" className="absolute inset-0 h-full w-full touch-none" onPointerMove={onPointerMove} onPointerUp={stopDrag} onPointerLeave={stopDrag}>
              {previewUrl ? (
                <image href={previewUrl} x="0" y="0" width="125" height="92" preserveAspectRatio="none" opacity="0.58" />
              ) : (
                <rect x="3" y="3" width="116" height="84" rx="2" fill="rgba(255,255,255,0.14)" stroke="#334155" strokeWidth="1.2" />
              )}
              {rooms.map((room, index) => {
                const selected = selectedId === room.id;
                const wallStroke = 0.8 + Number(room.border_thickness_ft || 0.5) * 1.4 + (selected ? 0.8 : 0);
                const polygon = polygonFromRoom(room);
                const box = bboxFromPolygon(polygon);
                const points = polygon.map(([x, y]) => `${x},${y}`).join(" ");
                const label = labelPositionFromRoom(room);
                const resizeHandles = [
                  ["nw", box.minX, box.minY],
                  ["n", box.minX + box.width / 2, box.minY],
                  ["ne", box.maxX, box.minY],
                  ["e", box.maxX, box.minY + box.height / 2],
                  ["se", box.maxX, box.maxY],
                  ["s", box.minX + box.width / 2, box.maxY],
                  ["sw", box.minX, box.maxY],
                  ["w", box.minX, box.minY + box.height / 2],
                ];
                return (
                  <g key={room.id}>
                    <polygon
                      points={points}
                      fill={COLORS[index % COLORS.length]}
                      fillOpacity={selected ? "0.30" : "0.15"}
                      stroke={selected ? "#111827" : COLORS[index % COLORS.length]}
                      strokeWidth={wallStroke}
                      className="cursor-move"
                      onPointerDown={(event) => onPointerDown(event, room)}
                    />
                    {selected ? (
                      <rect
                        x={box.minX}
                        y={box.minY}
                        width={box.width}
                        height={box.height}
                        fill="none"
                        stroke="#4F46E5"
                        strokeWidth="0.65"
                        strokeDasharray="1.5 1.5"
                        pointerEvents="none"
                      />
                    ) : null}
                    <text x={label.x} y={label.y} fill="#111827" fontSize="3.1" fontWeight="700" pointerEvents="none">
                      {room.label || "Pending AI label"}
                    </text>
                    <text x={label.x} y={label.areaY} fill="#475569" fontSize="2.5" pointerEvents="none">
                      {room.border_thickness_ft || 0.5} ft wall
                    </text>
                    {selected ? polygon.map(([x, y], edgeIndex) => {
                      const next = polygon[(edgeIndex + 1) % polygon.length];
                      return (
                        <circle
                          key={`${room.id}-edge-${edgeIndex}`}
                          cx={(x + next[0]) / 2}
                          cy={(y + next[1]) / 2}
                          r="1.45"
                          fill="#ffffff"
                          stroke="#4F46E5"
                          strokeWidth="0.8"
                          className="cursor-crosshair"
                          onPointerDown={(event) => onEdgePointerDown(event, room, edgeIndex)}
                        />
                      );
                    }) : null}
                    {selected ? resizeHandles.map(([handle, x, y]) => (
                      <rect
                        key={`${room.id}-resize-${handle}`}
                        x={x - 1.65}
                        y={y - 1.65}
                        width="3.3"
                        height="3.3"
                        rx="0.55"
                        fill="#ffffff"
                        stroke="#111827"
                        strokeWidth="0.7"
                        className={handle.includes("n") && handle.includes("w") ? "cursor-nwse-resize" : handle.includes("n") && handle.includes("e") ? "cursor-nesw-resize" : handle.includes("s") && handle.includes("e") ? "cursor-nwse-resize" : handle.includes("s") && handle.includes("w") ? "cursor-nesw-resize" : handle.includes("e") || handle.includes("w") ? "cursor-ew-resize" : "cursor-ns-resize"}
                        onPointerDown={(event) => onResizePointerDown(event, room, handle)}
                      />
                    )) : null}
                    {selected ? polygon.map(([x, y], vertexIndex) => (
                      <circle
                        key={`${room.id}-vertex-${vertexIndex}`}
                        cx={x}
                        cy={y}
                        r={selectedVertex === vertexIndex ? 2.35 : 1.85}
                        fill={selectedVertex === vertexIndex ? "#4F46E5" : "#111827"}
                        stroke="#ffffff"
                        strokeWidth="0.55"
                        className="cursor-grab"
                        onPointerDown={(event) => onPointerDown(event, room, "vertex", vertexIndex)}
                      />
                    )) : null}
                  </g>
                );
              })}
            </svg>
            <div className="absolute bottom-4 left-4 flex items-center gap-2 rounded-xl border border-slate-200 bg-white/92 px-3 py-2 text-xs font-semibold text-[#475569] shadow-sm backdrop-blur">
              <MousePointer2 className="h-4 w-4 text-[#4F46E5]" /> Drag black corners to reshape. Drag square handles to resize. Drag white edge dots to create new corners.
            </div>
          </div>
        </main>

        <aside className="min-h-0 overflow-y-auto border-l border-slate-200 bg-white">
          <div className="border-b border-slate-200 px-4 py-3">
            <div className="flex items-center gap-2">
              <SlidersHorizontal className="h-4 w-4 text-[#4F46E5]" />
              <p className="text-sm font-bold">Selected Zone</p>
            </div>
            <p className="mt-1 text-xs text-[#64748B]">Fix boundaries, resize, and correct labels only where AI is wrong.</p>
          </div>
          {selectedRoom ? (
            <div className="space-y-4 p-4">
              <div className="rounded-xl border border-slate-200 bg-slate-50 p-3">
                <p className="text-sm font-bold text-[#111827]">{selectedRoom.label || "Pending AI label"}</p>
                <p className="mt-1 text-xs text-[#64748B]">{selectedRoom.user_corrected_label ? "User-corrected label" : "AI predicted label"}</p>
              </div>
              <label className="block text-sm font-semibold text-[#475569]">
                Zone type
                <select
                  className="input-glass mt-2 w-full rounded-md px-3 py-2"
                  value={selectedRoom.type || "unlabeled"}
                  onChange={(event) => {
                    const selected = zoneTypes.find(([value]) => value === event.target.value);
                    updateRoom(selectedRoom.id, {
                      type: event.target.value,
                      label: selectedRoom.label && selectedRoom.label !== "Pending AI label" ? selectedRoom.label : selected?.[1] || "Unlabeled",
                      user_corrected_label: true,
                    });
                  }}
                >
                  {zoneTypes.map(([value, label]) => <option key={value} value={value}>{label}</option>)}
                </select>
              </label>
              <label className="block text-sm font-semibold text-[#475569]">
                Zone name
                <input
                  type="text"
                  className="input-glass mt-2 w-full rounded-md px-3 py-2"
                  value={selectedRoom.label || ""}
                  onChange={(event) => updateRoom(selectedRoom.id, { label: event.target.value, user_corrected_label: true })}
                />
              </label>
              <label className="block text-sm font-semibold text-[#475569]">
                Border thickness (ft)
                <input type="number" min="0.25" max="2" step="0.05" className="input-glass mt-2 w-full rounded-md px-3 py-2" value={selectedRoom.border_thickness_ft || 0.5} onChange={(event) => updateRoom(selectedRoom.id, { border_thickness_ft: Number(event.target.value || 0.5) })} />
              </label>
              <div className="grid grid-cols-2 gap-3">
                {[
                  ["x", "X"],
                  ["y", "Y"],
                  ["width", "Width"],
                  ["height", "Height"],
                ].map(([key, label]) => (
                  <label key={key} className="text-sm font-semibold text-[#475569]">
                    {label}
                    <input type="number" step="0.5" className="input-glass mt-2 w-full rounded-md px-3 py-2" value={selectedRoom[key]} onChange={(event) => updateRoom(selectedRoom.id, { [key]: Number(event.target.value || 0) })} />
                  </label>
                ))}
              </div>
              <div className="grid grid-cols-2 gap-2">
                <Button type="button" variant="secondary" icon={ArrowUp} disabled={!selectedId || rooms[rooms.length - 1]?.id === selectedId} onClick={() => moveLayer(selectedId, 1)}>Bring Up</Button>
                <Button type="button" variant="secondary" icon={ArrowDown} disabled={!selectedId || rooms[0]?.id === selectedId} onClick={() => moveLayer(selectedId, -1)}>Send Down</Button>
              </div>
              <div className="grid grid-cols-2 gap-2">
                <Button type="button" variant="secondary" icon={Plus} onClick={addVertex}>Add Point</Button>
                <Button type="button" variant="secondary" icon={Trash2} disabled={selectedVertex === null || polygonFromRoom(selectedRoom).length <= 3} onClick={deleteVertex}>Delete Point</Button>
              </div>
              <Button
                type="button"
                variant="secondary"
                icon={Wand2}
                className="w-full"
                onClick={() => updateRoom(selectedRoom.id, { user_corrected_label: false, label: "Pending AI label", type: "unlabeled" })}
              >
                Let AI Predict This Zone
              </Button>
              <div className="rounded-xl bg-[#EEF2FF] p-3 text-xs font-semibold leading-5 text-[#3730A3]">
                Drag black corner dots inward/outward to make irregular shapes. Drag white edge dots to add points. Drag square handles to resize while keeping the custom shape.
              </div>
            </div>
          ) : (
            <div className="p-4 text-sm font-semibold text-[#64748B]">Select a zone from the canvas or layer stack.</div>
          )}
        </aside>
      </div>
      <div className="border-t border-slate-200 bg-white px-5 py-3 text-xs font-semibold text-[#64748B]">
        {rooms.length} editable layer(s). User-corrected labels are preserved; unlocked zones can be rechecked by AI.
      </div>
    </div>
  );
}

function Signal({ label, value }) {
  return (
    <div className="rounded-xl border border-slate-200 bg-white/80 p-3">
      <div className="flex justify-between gap-3 text-xs font-bold uppercase text-[#64748B]">
        <span>{label}</span>
        <span>{value}%</span>
      </div>
      <div className="mt-2 h-2 overflow-hidden rounded-full bg-slate-100">
        <div className="h-full rounded-full bg-[#4F46E5]" style={{ width: `${value}%` }} />
      </div>
    </div>
  );
}

function StepPill({ index, active, done, label }) {
  return (
    <div className={`flex items-center gap-2 rounded-full border px-3 py-2 text-xs font-bold ${active ? "border-[#4F46E5] bg-[#EEF2FF] text-[#3730A3]" : done ? "border-emerald-200 bg-emerald-50 text-emerald-700" : "border-slate-200 bg-white text-[#64748B]"}`}>
      {done ? <CheckCircle2 className="h-4 w-4" /> : <span className="flex h-5 w-5 items-center justify-center rounded-full bg-slate-100">{index}</span>}
      {label}
    </div>
  );
}

export default function FlatWiseEstimator({ project }) {
  const [step, setStep] = useState(1);
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [upload, setUpload] = useState(null);
  const [detection, setDetection] = useState(null);
  const [editedRooms, setEditedRooms] = useState([]);
  const [originalRooms, setOriginalRooms] = useState([]);
  const [selectedZoneId, setSelectedZoneId] = useState(null);
  const [editedSummary, setEditedSummary] = useState(null);
  const [originalSummary, setOriginalSummary] = useState(null);
  const [flatRate, setFlatRate] = useState(project.estimate?.rate || 2200);
  const [tier, setTier] = useState("Standard");
  const [estimate, setEstimate] = useState(null);
  const [loading, setLoading] = useState("");
  const [error, setError] = useState("");
  const [reprocessAttempt, setReprocessAttempt] = useState(0);
  const [editorOpen, setEditorOpen] = useState(false);
  const isConfirmStep = step === 3 && detection;

  const corrections = useMemo(() => {
    if (!editedSummary || !originalSummary) return 0;
    const fieldCorrections = editableFields.filter(([key]) => String(editedSummary[key]) !== String(originalSummary[key])).length;
    const zoneCorrections = JSON.stringify(editedRooms) === JSON.stringify(originalRooms) ? 0 : 1;
    return fieldCorrections + zoneCorrections;
  }, [editedSummary, originalSummary, editedRooms, originalRooms]);

  const goBack = () => {
    setError("");
    setStep((current) => Math.max(1, current - 1));
  };

  const uploadFile = async (demoSample = false) => {
    setError("");
    setLoading(demoSample ? "demo" : "upload");
    try {
      const response = await endpoints.uploadPlan({ projectId: project.id, file, demoSample });
      setUpload(response);
      setDetection(null);
      setEditedRooms([]);
      setOriginalRooms([]);
      setSelectedZoneId(null);
      setEditorOpen(false);
      setEstimate(null);
      setReprocessAttempt(0);
      if (response.preview_data_url) setPreviewUrl(response.preview_data_url);
      if (demoSample) setPreviewUrl(null);
      setStep(2);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading("");
    }
  };

  const processLayout = async (isReprocess = false) => {
    setError("");
    if (!upload?.plan_id) {
      setError("Upload a floorplan or use the demo sample first.");
      return;
    }
    const nextAttempt = isReprocess ? reprocessAttempt + 1 : reprocessAttempt;
    setLoading("detect");
    try {
      const response = await endpoints.detectLayout({ project_id: project.id, plan_id: upload.plan_id, reprocess_attempt: nextAttempt });
      setDetection(response);
      setEditedRooms(response.rooms || []);
      setOriginalRooms(response.rooms || []);
      setSelectedZoneId(response.rooms?.[0]?.id || null);
      setEditedSummary(response.summary);
      setOriginalSummary(response.summary);
      setReprocessAttempt(nextAttempt);
      setEstimate(null);
      setStep(3);
      setEditorOpen(false);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading("");
    }
  };

  const autoLabelEditedZones = async () => {
    setError("");
    if (!upload?.plan_id || !editedRooms.length) {
      setError("Edit or add at least one zone before auto-labeling.");
      return;
    }
    setLoading("relabel");
    try {
      const response = await endpoints.relabelZones({
        project_id: project.id,
        plan_id: upload.plan_id,
        rooms: editedRooms,
        summary: editedSummary,
        user_corrections: corrections || 1,
      });
      setDetection(response);
      setEditedRooms(response.rooms || []);
      setOriginalRooms(response.rooms || []);
      setSelectedZoneId(response.rooms?.[0]?.id || null);
      setEditedSummary(response.summary);
      setOriginalSummary(response.summary);
      setEstimate(null);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading("");
    }
  };

  const confirm = async () => {
    setError("");
    setLoading("confirm");
    try {
      const response = await endpoints.confirmLayout({
        project_id: project.id,
        plan_id: upload.plan_id,
        summary: editedSummary,
        rooms: editedRooms,
        user_corrections: corrections,
      });
      setDetection(response);
      setEditedRooms(response.rooms || []);
      setOriginalRooms(response.rooms || []);
      setEditedSummary(response.summary);
      setOriginalSummary(response.summary);
      setStep(4);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading("");
    }
  };

  const estimateFlat = async () => {
    setError("");
    if (!Number(flatRate) || Number(flatRate) <= 0) {
      setError("Enter a valid construction rate per sq.ft before generating the estimate.");
      return;
    }
    setLoading("estimate");
    try {
      if (detection && editedSummary) {
        const confirmed = await endpoints.confirmLayout({
          project_id: project.id,
          plan_id: upload.plan_id,
          summary: editedSummary,
          rooms: editedRooms,
          user_corrections: corrections,
        });
        setDetection(confirmed);
        setEditedRooms(confirmed.rooms || []);
        setOriginalRooms(confirmed.rooms || []);
        setEditedSummary(confirmed.summary);
        setOriginalSummary(confirmed.summary);
      }
      const response = await endpoints.estimateFlat({
        project_id: project.id,
        plan_id: upload.plan_id,
        rate_per_sqft: Number(flatRate),
        tier,
      });
      setEstimate(response);
      setStep(5);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading("");
    }
  };

  const chooseFile = (event) => {
    const selected = event.target.files?.[0];
    setFile(selected || null);
    setUpload(null);
    setDetection(null);
    setEditedRooms([]);
    setOriginalRooms([]);
    setSelectedZoneId(null);
    setEditorOpen(false);
    setEstimate(null);
    setReprocessAttempt(0);
    setStep(1);
    if (selected && selected.type.startsWith("image/")) {
      setPreviewUrl(URL.createObjectURL(selected));
    } else {
      setPreviewUrl(null);
    }
  };

  return (
    <section className="glass-panel p-4 sm:p-5">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <p className="text-sm font-semibold uppercase tracking-wide text-[#4F46E5]">Flat Wise AI estimation</p>
          <h2 className="mt-1 text-2xl font-bold text-[#111827]">Plan upload to material estimate</h2>
          <p className="mt-2 max-w-3xl text-sm leading-6 text-[#64748B]">
            Upload a flat plan, let the demo parser detect rooms and geometry, confirm only uncertain values, then generate area and material-based estimates.
          </p>
        </div>
        <div className="flex flex-wrap gap-2">
          {step > 1 ? (
            <Button type="button" variant="secondary" icon={ArrowLeft} onClick={goBack}>
              Back
            </Button>
          ) : null}
          <StepPill index={1} active={step === 1} done={step > 1} label="Upload" />
          <StepPill index={2} active={step === 2} done={step > 2} label="Detect" />
          <StepPill index={3} active={step === 3} done={step > 3} label="Confirm" />
          <StepPill index={4} active={step === 4} done={step > 4} label="Rate" />
          <StepPill index={5} active={step === 5} done={false} label="Dashboard" />
        </div>
      </div>

      {error ? <div className="mt-4 rounded-xl border border-red-200 bg-red-50 p-3 text-sm font-semibold text-red-700">{error}</div> : null}

      {editorOpen && step === 3 && detection ? (
        <ZoneEditor
          rooms={editedRooms}
          setRooms={setEditedRooms}
          previewUrl={previewUrl}
          selectedId={selectedZoneId}
          setSelectedId={setSelectedZoneId}
          onAutoLabel={autoLabelEditedZones}
          loading={loading}
          onClose={() => setEditorOpen(false)}
        />
      ) : null}

      <div className={`mt-4 grid gap-4 ${isConfirmStep ? "xl:grid-cols-1" : "xl:grid-cols-[1.05fr_0.95fr]"}`}>
        <div className="space-y-3">
          <DetectionPreview detection={detection} previewUrl={previewUrl} imageSize={upload?.image_size} />
          {step === 3 && detection ? (
            <div className="rounded-2xl border border-slate-200 bg-white/86 p-3 shadow-sm">
              <div className="flex flex-wrap items-center justify-between gap-3">
                <div>
                  <h3 className="font-bold text-[#111827]">Full-screen zone editor</h3>
                  <p className="mt-1 text-sm text-[#64748B]">Open the canvas to edit boundaries, resize zones, correct labels, tune wall thickness, and reorder layers.</p>
                </div>
                <Button type="button" icon={Maximize2} onClick={() => setEditorOpen(true)}>
                  Open Full Screen Canvas
                </Button>
              </div>
            </div>
          ) : null}
          {detection ? (
            <div className="grid gap-2 sm:grid-cols-3 lg:grid-cols-6">
              <Signal label="OCR" value={detection.signals.ocr_confidence} />
              <Signal label="Polygons" value={detection.signals.polygon_completeness} />
              <Signal label="Geometry" value={detection.signals.geometry_consistency} />
              <Signal label="Dimensions" value={detection.signals.dimension_confidence || 0} />
              <Signal label="Image" value={detection.signals.image_quality} />
              <div className="rounded-xl border border-slate-200 bg-white/80 p-3">
                <p className="text-xs font-bold uppercase text-[#64748B]">OCR tokens</p>
                <p className="mt-2 text-lg font-bold text-[#111827]">{detection.ocr?.word_count || 0}</p>
                <p className="text-xs text-[#64748B]">{detection.ocr?.avg_confidence || 0}% avg</p>
              </div>
            </div>
          ) : null}
        </div>

        <div className="space-y-3">
          {step === 1 ? (
          <div className="rounded-2xl border border-slate-200 bg-white/82 p-4 shadow-sm">
            <div className="flex items-center gap-2">
              <UploadCloud className="h-5 w-5 text-[#4F46E5]" />
              <h3 className="font-bold text-[#111827]">Step 1 - Upload floorplan</h3>
            </div>
            <label className="mt-4 flex min-h-28 cursor-pointer flex-col items-center justify-center rounded-2xl border-2 border-dashed border-slate-300 bg-slate-50 px-4 py-6 text-center transition hover:border-[#4F46E5] hover:bg-[#EEF2FF]">
              <FileUp className="h-6 w-6 text-[#4F46E5]" />
              <span className="mt-2 text-sm font-bold text-[#111827]">{file ? file.name : "Upload PNG, JPG, JPEG, DXF, or PDF"}</span>
              <span className="mt-1 text-xs text-[#64748B]">The demo uses OCR, contour, polygon, and CAD-label heuristics.</span>
              <input type="file" accept=".png,.jpg,.jpeg,.dxf,.pdf" className="hidden" onChange={chooseFile} />
            </label>
            <div className="mt-4 flex flex-wrap gap-2">
              <Button type="button" icon={UploadCloud} loading={loading === "upload"} disabled={!file} onClick={() => uploadFile(false)}>
                Upload File
              </Button>
              <Button type="button" variant="secondary" icon={Sparkles} loading={loading === "demo"} onClick={() => uploadFile(true)}>
                Try Demo Sample
              </Button>
            </div>
            {upload ? <p className="mt-3 text-sm font-semibold text-emerald-700">{upload.message} Quality score: {upload.quality_score}%.</p> : null}
          </div>
          ) : null}

          {step === 2 ? (
            <div className="rounded-2xl border border-slate-200 bg-white/82 p-4 shadow-sm">
              <div className="flex items-center gap-2">
                <BrainCircuit className="h-5 w-5 text-[#4F46E5]" />
                <h3 className="font-bold text-[#111827]">Step 2 - AI-assisted detection</h3>
              </div>
              <p className="mt-3 text-sm leading-6 text-[#64748B]">
                Uploaded file is ready. The engine will run wall contour segmentation, line-grid analysis, label hints, area checks, and confidence scoring.
              </p>
              {upload ? (
                <div className="mt-4 rounded-xl bg-slate-50 p-3 text-sm font-semibold text-[#475569]">
                  {upload.file_name} | Quality score {upload.quality_score}% | {upload.file_type.toUpperCase()}
                </div>
              ) : null}
              <div className="mt-4 flex flex-wrap gap-2">
                <Button type="button" icon={BrainCircuit} loading={loading === "detect"} disabled={!upload} onClick={() => processLayout(false)}>
                  Process Layout
                </Button>
                <Button type="button" variant="secondary" icon={RefreshCw} loading={loading === "detect"} disabled={!upload} onClick={() => processLayout(true)}>
                  Alternate Pass
                </Button>
              </div>
            </div>
          ) : null}

          {step === 3 && detection ? (
            <div className="rounded-2xl border border-slate-200 bg-white/82 p-3 shadow-sm sm:p-4">
              <div className="flex items-start justify-between gap-3">
                <div>
                  <h3 className="font-bold text-[#111827]">Step 3 - Confirm detected layout</h3>
                  <p className="mt-1 text-sm text-[#64748B]">Only meaningful fields are editable. Corrections reduce confidence slightly.</p>
                </div>
                <div className="rounded-xl bg-[#EEF2FF] px-3 py-2 text-sm font-bold text-[#3730A3]">{detection.confidence}% confidence</div>
              </div>
              {detection.geometry_reliability?.display_warning ? (
                <div className="mt-3 rounded-xl border border-amber-200 bg-amber-50 p-3 text-sm font-semibold leading-6 text-amber-800">
                  {detection.geometry_reliability.display_warning}
                </div>
              ) : null}
              <div className="mt-3 grid gap-2 sm:grid-cols-2 xl:grid-cols-3">
                {editableFields.map(([key, label]) => (
                  <label key={key} className="text-xs font-bold uppercase tracking-wide text-[#64748B]">
                    {label}
                    <input
                      type="number"
                      className="input-glass mt-1.5 w-full rounded-md px-3 py-2 text-sm font-semibold normal-case tracking-normal text-[#111827]"
                      value={editedSummary?.[key] ?? ""}
                      onChange={(event) => setEditedSummary({ ...editedSummary, [key]: Number(event.target.value) })}
                    />
                  </label>
                ))}
              </div>
              <div className="mt-3 grid gap-2 md:grid-cols-4">
                <div className="rounded-xl border border-slate-200 bg-white p-2.5">
                  <p className="text-xs font-bold uppercase text-[#64748B]">Primary rooms</p>
                  <p className="mt-1 text-xl font-bold text-[#111827]">{editedSummary?.room_count ?? 0}</p>
                </div>
                <div className="rounded-xl border border-slate-200 bg-white p-2.5">
                  <p className="text-xs font-bold uppercase text-[#64748B]">Raw zones</p>
                  <p className="mt-1 text-xl font-bold text-[#111827]">{detection.summary?.raw_zone_count ?? detection.signals?.segmented_zone_count ?? 0}</p>
                </div>
                <div className="rounded-xl border border-slate-200 bg-white p-2.5">
                  <p className="text-xs font-bold uppercase text-[#64748B]">Wet areas</p>
                  <p className="mt-1 text-xl font-bold text-[#111827]">{detection.summary?.wet_area_count ?? editedSummary?.bathroom_count ?? 0}</p>
                </div>
                <div className="rounded-xl border border-slate-200 bg-white p-2.5">
                  <p className="text-xs font-bold uppercase text-[#64748B]">Source</p>
                  <p className="mt-1 text-sm font-bold capitalize text-[#111827]">{String(detection.summary?.count_source || "AI fusion").replaceAll("-", " ")}</p>
                </div>
              </div>
              <div className="mt-3 grid gap-2 md:grid-cols-3">
                <div className="rounded-xl bg-slate-50 p-2.5">
                  <p className="text-xs font-bold uppercase text-[#64748B]">Geometry mode</p>
                  <p className="mt-1 text-sm font-bold capitalize text-[#111827]">{String(detection.summary?.layout_geometry_mode || "unknown").replaceAll("-", " ")}</p>
                </div>
                <div className="rounded-xl bg-slate-50 p-2.5">
                  <p className="text-xs font-bold uppercase text-[#64748B]">Geometry confidence</p>
                  <p className="mt-1 text-sm font-bold capitalize text-[#111827]">{detection.summary?.layout_geometry_confidence || "review"}</p>
                </div>
                <div className="rounded-xl bg-slate-50 p-2.5">
                  <p className="text-xs font-bold uppercase text-[#64748B]">Fallbacks used</p>
                  <p className="mt-1 text-sm font-bold text-[#111827]">{detection.signals?.fallback_count || 0}</p>
                </div>
              </div>
              <div className="mt-3 rounded-xl bg-slate-50 p-3 text-sm leading-6 text-[#475569]">
                <strong className="text-[#111827]">Is this correct?</strong> Detected {editedSummary?.room_count} primary rooms from {detection.summary?.raw_zone_count ?? detection.signals?.segmented_zone_count ?? 0} raw zones, including {editedSummary?.kitchen_count} kitchen and {editedSummary?.bathroom_count} bathrooms. Wall thickness is around {editedSummary?.wall_thickness_ft} ft, and built-up area is around {Math.round(editedSummary?.built_up_area_sqft || 0)} sq.ft.
              </div>
              <div className="mt-3 rounded-2xl border border-slate-200 bg-white p-3">
                <div className="flex items-center justify-between gap-3">
                  <div>
                    <h4 className="font-bold text-[#111827]">Pattern-based quantity takeoff</h4>
                    <p className="mt-1 text-xs font-semibold text-[#64748B]">Cost engine prioritizes dimensions, wall length, wall area, carpet area, and openings over room count.</p>
                  </div>
                  <span className="rounded-md bg-[#EEF2FF] px-2 py-1 text-xs font-bold text-[#3730A3]">
                    {detection.signals?.takeoff_confidence || 0}% takeoff
                  </span>
                </div>
                <div className="mt-3 grid gap-2 sm:grid-cols-3 xl:grid-cols-6">
                  {[
                    ["Total wall length", `${Math.round(editedSummary?.total_wall_length_ft || 0)} ft`],
                    ["Net wall area", `${Math.round(editedSummary?.net_wall_area_sqft || 0)} sqft`],
                    ["Tiles area", `${Math.round(editedSummary?.tile_area_sqft || 0)} sqft`],
                    ["Brick count", `${Math.round(editedSummary?.brick_count || 0).toLocaleString("en-IN")} nos`],
                    ["Concrete", `${Number(editedSummary?.concrete_volume_cum || 0).toFixed(1)} cum`],
                    ["Dimension source", String(editedSummary?.dimension_source || "area-derived").replace("-", " ")],
                  ].map(([label, value]) => (
                    <div key={label} className="rounded-xl bg-slate-50 p-2.5">
                      <p className="text-[11px] font-bold uppercase text-[#64748B]">{label}</p>
                      <p className="mt-1 text-sm font-bold text-[#111827]">{value}</p>
                    </div>
                  ))}
                </div>
              </div>
              <div className="mt-3 flex flex-wrap gap-2">
                <Button type="button" icon={BadgeCheck} loading={loading === "confirm"} onClick={confirm}>
                  Confirm Layout
                </Button>
                <Button type="button" variant="secondary" icon={RefreshCw} loading={loading === "detect"} onClick={() => processLayout(true)}>
                  Reject / Reprocess {reprocessAttempt ? `(${reprocessAttempt + 1})` : ""}
                </Button>
              </div>
            </div>
          ) : null}

          {step === 4 && detection ? (
            <div className="rounded-2xl border border-slate-200 bg-white/82 p-4 shadow-sm">
              <h3 className="font-bold text-[#111827]">Step 4 - Enter construction rate</h3>
              <div className="mt-4 grid gap-3 sm:grid-cols-[1fr_0.8fr]">
                <label className="text-sm font-semibold text-[#475569]">
                  Construction cost per sq.ft
                  <input type="number" min="1" className="input-glass mt-2 w-full rounded-md px-3 py-3" value={flatRate} onChange={(event) => setFlatRate(event.target.value)} />
                </label>
                <label className="text-sm font-semibold text-[#475569]">
                  Finish tier
                  <select className="input-glass mt-2 w-full rounded-md px-3 py-3" value={tier} onChange={(event) => setTier(event.target.value)}>
                    <option>Standard</option>
                    <option>Premium</option>
                    <option>Luxury</option>
                  </select>
                </label>
              </div>
              <Button type="button" className="mt-4" icon={Home} loading={loading === "estimate"} onClick={estimateFlat}>
                Generate Flat Estimate
              </Button>
            </div>
          ) : null}
        </div>
      </div>

      {detection && step < 5 ? (
        <div className="mt-5 grid gap-4 lg:grid-cols-[0.9fr_1.1fr]">
          <div className="rounded-2xl border border-slate-200 bg-white/82 p-4">
            <h3 className="flex items-center gap-2 font-bold text-[#111827]"><BrainCircuit className="h-5 w-5 text-[#4F46E5]" /> Confidence engine</h3>
            <p className="mt-3 text-sm leading-6 text-[#475569]">{detection.confidence_explanation}</p>
            <div className="mt-3 grid gap-2 sm:grid-cols-4">
              <Signal label="OCR" value={detection.signals?.ocr_confidence || 0} />
              <Signal label="Geometry" value={detection.signals?.geometry_consistency || 0} />
              <Signal label="Polygons" value={detection.signals?.polygon_completeness || 0} />
              <Signal label="Dimensions" value={detection.signals?.dimension_confidence || 0} />
            </div>
            <div className="mt-3 rounded-xl bg-[#EEF2FF] p-3 text-xs font-semibold leading-5 text-[#3730A3]">
              The app now separates raw detected zones from actual primary rooms. Raw zones include fragments, furniture blocks, balconies, wet areas, and open geometry; they are not used as final room count without OCR confirmation.
            </div>
            {detection.processing_trace ? (
              <div className="mt-3 rounded-xl border border-slate-200 bg-white p-3 text-xs leading-5 text-[#475569]">
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <span><strong className="text-[#111827]">Pipeline:</strong> {detection.processing_trace.path?.join(" -> ") || "not recorded"}</span>
                  <span className="rounded-md bg-slate-100 px-2 py-1 font-bold text-[#334155]">{detection.processing_trace.pipeline_version}</span>
                </div>
                <div className="mt-2 grid gap-2 sm:grid-cols-3">
                  <span><strong className="text-[#111827]">Deterministic:</strong> {detection.processing_trace.deterministic ? "Yes" : "Alternate pass"}</span>
                  <span><strong className="text-[#111827]">Fallbacks:</strong> {(detection.processing_trace.fallbacks_used || []).join(", ") || "None"}</span>
                  <span><strong className="text-[#111827]">Input hash:</strong> {String(detection.processing_trace.input_hash || "").slice(0, 10) || "n/a"}</span>
                </div>
              </div>
            ) : null}
            {detection.ocr?.text ? (
              <div className="mt-3 rounded-xl bg-slate-50 p-3 text-xs leading-5 text-[#475569]">
                <span className="font-bold text-[#111827]">OCR text:</span> {detection.ocr.text}
              </div>
            ) : null}
            <div className="mt-4 h-3 overflow-hidden rounded-full bg-slate-100">
              <div className="h-full rounded-full bg-gradient-to-r from-[#4F46E5] to-[#0891B2]" style={{ width: `${detection.confidence}%` }} />
            </div>
          </div>
          <div className="rounded-2xl border border-slate-200 bg-white/82 p-4">
            <h3 className="flex items-center gap-2 font-bold text-[#111827]"><AlertTriangle className="h-5 w-5 text-[#D97706]" /> Technical advisor</h3>
            <div className="mt-3 grid gap-2">
              {detection.recommendations.map((item, index) => (
                <div key={`${index}-${item}`} className="rounded-xl bg-slate-50 p-3 text-sm font-medium text-[#475569]">{item}</div>
              ))}
            </div>
          </div>
        </div>
      ) : null}

      {estimate && step === 5 ? (
        <div className="mt-5 space-y-5">
          <div className="grid gap-3 md:grid-cols-4">
            <div className="glass-panel p-4"><p className="text-xs font-bold uppercase text-[#64748B]">Total estimate</p><p className="mt-2 text-2xl font-bold">{money(estimate.total_estimated_cost)}</p></div>
            <div className="glass-panel p-4"><p className="text-xs font-bold uppercase text-[#64748B]">Rate / sqft</p><p className="mt-2 text-2xl font-bold">{money(estimate.rate_per_sqft)}</p></div>
            <div className="glass-panel p-4"><p className="text-xs font-bold uppercase text-[#64748B]">Material cost</p><p className="mt-2 text-2xl font-bold">{shortMoney(estimate.estimated_material_cost)}</p></div>
            <div className="glass-panel p-4"><p className="text-xs font-bold uppercase text-[#64748B]">Confidence</p><p className="mt-2 text-2xl font-bold">{estimate.confidence}%</p></div>
          </div>

          <div className="grid gap-5 xl:grid-cols-2">
            <div className="rounded-2xl border border-slate-200 bg-white/86 p-5">
              <h3 className="font-bold">Area vs material model</h3>
              <div className="mt-4 grid gap-3 sm:grid-cols-3">
                <div className="rounded-xl bg-slate-50 p-3"><p className="text-xs font-bold uppercase text-[#64748B]">Area estimate</p><p className="mt-1 font-bold">{shortMoney(estimate.area_based_estimate)}</p></div>
                <div className="rounded-xl bg-slate-50 p-3"><p className="text-xs font-bold uppercase text-[#64748B]">Material estimate</p><p className="mt-1 font-bold">{shortMoney(estimate.material_based_estimate)}</p></div>
                <div className="rounded-xl bg-slate-50 p-3"><p className="text-xs font-bold uppercase text-[#64748B]">Variance</p><p className="mt-1 font-bold">{estimate.variance_percent}%</p></div>
              </div>
              <p className="mt-4 rounded-xl bg-[#EEF2FF] p-3 text-sm font-semibold text-[#3730A3]">{estimate.confidence_explanation}</p>
            </div>
            <div className="rounded-2xl border border-slate-200 bg-white/86 p-5">
              <h3 className="font-bold">Dimension and wall takeoff</h3>
              <div className="mt-4 grid gap-3 sm:grid-cols-2">
                {[
                  ["Built-up area", `${Math.round(estimate.area_details?.built_up_area_sqft || 0)} sqft`],
                  ["Carpet area", `${Math.round(estimate.area_details?.carpet_area_sqft || 0)} sqft`],
                  ["Wall length", `${Math.round(estimate.area_details?.total_wall_length_ft || 0)} ft`],
                  ["Wall area", `${Math.round(estimate.area_details?.net_wall_area_sqft || 0)} sqft`],
                  ["Brick count", `${Math.round(estimate.area_details?.brick_count || 0).toLocaleString("en-IN")} nos`],
                  ["Concrete", `${Number(estimate.area_details?.concrete_volume_cum || 0).toFixed(1)} cum`],
                ].map(([label, value]) => (
                  <div key={label} className="rounded-xl bg-slate-50 p-3">
                    <p className="text-xs font-bold uppercase text-[#64748B]">{label}</p>
                    <p className="mt-1 font-bold text-[#111827]">{value}</p>
                  </div>
                ))}
              </div>
            </div>
          </div>

          <div className="grid gap-5 xl:grid-cols-2">
            <div className="rounded-2xl border border-slate-200 bg-white/86 p-5">
              <h3 className="font-bold">Cost distribution</h3>
              <div className="mt-4 h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <PieChart>
                    <Pie data={estimate.cost_distribution} dataKey="value" nameKey="name" innerRadius={58} outerRadius={90}>
                      {estimate.cost_distribution.map((entry, index) => <Cell key={entry.name} fill={COLORS[index % COLORS.length]} />)}
                    </Pie>
                    <Tooltip formatter={(value) => money(value)} />
                  </PieChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>

          <div className="grid gap-5 xl:grid-cols-[1.1fr_0.9fr]">
            <div className="rounded-2xl border border-slate-200 bg-white/86 p-5">
              <h3 className="font-bold">Material breakdown</h3>
              <div className="mt-4 overflow-x-auto">
                <table className="w-full min-w-[760px] text-left text-sm">
                  <thead className="bg-slate-100 text-xs uppercase text-[#64748B]">
                    <tr><th className="px-3 py-3">Material</th><th className="px-3 py-3">Quantity</th><th className="px-3 py-3">Unit</th><th className="px-3 py-3">Waste</th><th className="px-3 py-3">Approx cost</th></tr>
                  </thead>
                  <tbody>
                    {estimate.material_breakdown.map((row) => (
                      <tr key={row.material} className="border-t border-slate-100">
                        <td className="px-3 py-3 font-bold">{row.material}</td>
                        <td className="px-3 py-3">{Number(row.quantity).toLocaleString("en-IN")}</td>
                        <td className="px-3 py-3">{row.unit}</td>
                        <td className="px-3 py-3">{Math.round(Number(row.waste_factor || 0) * 100)}%</td>
                        <td className="px-3 py-3 font-bold">{money(row.approx_cost)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
            <div className="rounded-2xl border border-slate-200 bg-white/86 p-5">
              <h3 className="font-bold">Material distribution</h3>
              <div className="mt-4 h-72">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={estimate.material_breakdown}>
                    <XAxis dataKey="material" />
                    <YAxis tickFormatter={(value) => shortMoney(value)} />
                    <Tooltip formatter={(value) => money(value)} />
                    <Bar dataKey="approx_cost" radius={[8, 8, 0, 0]}>
                      {estimate.material_breakdown.map((entry, index) => <Cell key={entry.material} fill={COLORS[index % COLORS.length]} />)}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>

          <div className="rounded-2xl border border-slate-200 bg-white/86 p-5">
            <h3 className="flex items-center gap-2 font-bold"><AlertTriangle className="h-5 w-5 text-[#D97706]" /> Final technical recommendations</h3>
            <div className="mt-3 grid gap-2 md:grid-cols-2">
              {estimate.recommendations.map((item, index) => <div key={`${index}-${item}`} className="rounded-xl bg-slate-50 p-3 text-sm font-medium text-[#475569]">{item}</div>)}
            </div>
          </div>
        </div>
      ) : null}

      {loading ? (
        <div className="fixed bottom-5 right-5 z-50 flex items-center gap-2 rounded-full bg-[#111827] px-4 py-3 text-sm font-bold text-white shadow-lg">
          <Loader2 className="h-4 w-4 animate-spin" /> Working on {loading}...
        </div>
      ) : null}
    </section>
  );
}

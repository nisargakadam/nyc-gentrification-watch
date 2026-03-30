// frontend/src/components/MapView.tsx
// Leaflet map — renders neighborhood risk markers

import { useEffect } from "react";
import { MapContainer, TileLayer, CircleMarker, Tooltip, useMap } from "react-leaflet";
import { riskColor } from "../pages/index";

interface Neighborhood {
  name: string;
  borough: string;
  lat: number;
  lng: number;
  risk_score: number;
  trend: string;
}

interface Props {
  neighborhoods: Neighborhood[];
  selected: Neighborhood | null;
  onSelect: (n: Neighborhood) => void;
}

// Recenter map when selection changes
function MapController({ selected }: { selected: Neighborhood | null }) {
  const map = useMap();
  useEffect(() => {
    if (selected) {
      map.flyTo([selected.lat, selected.lng], 14, { duration: 0.8 });
    }
  }, [selected, map]);
  return null;
}

export default function MapView({ neighborhoods, selected, onSelect }: Props) {
  return (
    <MapContainer
      center={[40.7128, -73.9660]}
      zoom={11}
      style={{ height: "100%", width: "100%", background: "#060d1a" }}
      zoomControl={false}
    >
      {/* Dark tile layer */}
      <TileLayer
        url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
        attribution='&copy; <a href="https://carto.com/">CARTO</a>'
      />

      <MapController selected={selected} />

      {neighborhoods.map((hood) => {
        const color   = riskColor(hood.risk_score);
        const isSelected = selected?.name === hood.name;
        const radius   = isSelected ? 20 : 14 + hood.risk_score * 12;

        return (
          <CircleMarker
            key={hood.name}
            center={[hood.lat, hood.lng]}
            radius={radius}
            pathOptions={{
              color:       isSelected ? "#ffffff" : color,
              fillColor:   color,
              fillOpacity: isSelected ? 0.9 : 0.6,
              weight:      isSelected ? 2.5 : 1,
            }}
            eventHandlers={{ click: () => onSelect(hood) }}
          >
            <Tooltip
              permanent={false}
              direction="top"
              offset={[0, -10]}
              className="leaflet-custom-tooltip"
            >
              <div style={{
                background: "#0a0f1e",
                border: `1px solid ${color}`,
                borderRadius: 6,
                padding: "6px 10px",
                color: "white",
                fontFamily: "DM Sans, sans-serif",
                fontSize: 12,
              }}>
                <strong>{hood.name}</strong>
                <div style={{ color, fontWeight: "bold" }}>
                  {Math.round(hood.risk_score * 100)} / 100
                </div>
                <div style={{ color: "#94a3b8", fontSize: 10 }}>{hood.trend}</div>
              </div>
            </Tooltip>
          </CircleMarker>
        );
      })}
    </MapContainer>
  );
}

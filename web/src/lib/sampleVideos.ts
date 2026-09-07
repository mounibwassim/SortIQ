/**
 * Sample Video & Canvas Generator for SortIQ Material Testing
 * Generates dynamic animated canvas video streams for Paper, Plastic, Metal, and Glass
 */

export interface MaterialSample {
  id: 'paper' | 'plastic' | 'metal' | 'glass';
  name: string;
  material: string;
  icon: string;
  color: string;
  description: string;
}

export const MATERIAL_SAMPLES: MaterialSample[] = [
  {
    id: 'paper',
    name: 'Paper & Cardboard',
    material: 'Paper',
    icon: '📄',
    color: '#f97316',
    description: 'Test recycling detection on paper boxes, cups, and documents',
  },
  {
    id: 'plastic',
    name: 'Plastic Container',
    material: 'Plastic',
    icon: '🥤',
    color: '#3b82f6',
    description: 'Test detection on plastic water bottles and food containers',
  },
  {
    id: 'metal',
    name: 'Metal Soda Can',
    material: 'Metal',
    icon: '🥫',
    color: '#eab308',
    description: 'Test detection on aluminum cans and tin packaging',
  },
  {
    id: 'glass',
    name: 'Glass Bottle / Jar',
    material: 'Glass',
    icon: '🍾',
    color: '#22c55e',
    description: 'Test detection on glass sauce jars and beverage bottles',
  },
];

export function renderSampleFrame(
  ctx: CanvasRenderingContext2D,
  width: number,
  height: number,
  materialId: string,
  timeMs: number
): void {
  ctx.save();
  ctx.clearRect(0, 0, width, height);

  // Background conveyor belt / table grid
  const bgGradient = ctx.createLinearGradient(0, 0, 0, height);
  bgGradient.addColorStop(0, '#0f172a');
  bgGradient.addColorStop(1, '#1e293b');
  ctx.fillStyle = bgGradient;
  ctx.fillRect(0, 0, width, height);

  // Animated grid lines (conveyor effect)
  const gridOffset = (timeMs / 20) % 40;
  ctx.strokeStyle = '#334155';
  ctx.lineWidth = 1;
  for (let y = gridOffset; y < height; y += 40) {
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(width, y);
    ctx.stroke();
  }

  // Object animation position (moves across screen)
  const cycleMs = 4000;
  const progress = (timeMs % cycleMs) / cycleMs;
  const cx = width * (0.2 + 0.6 * Math.sin(progress * Math.PI));
  const cy = height * 0.5 + Math.cos(progress * Math.PI * 2) * 15;

  ctx.translate(cx, cy);

  if (materialId === 'paper') {
    // 📄 Render Cardboard Box / Paper Stack
    ctx.fillStyle = '#d97706';
    ctx.beginPath();
    ctx.roundRect(-60, -45, 120, 90, 8);
    ctx.fill();

    ctx.fillStyle = '#f59e0b';
    ctx.beginPath();
    ctx.roundRect(-50, -35, 100, 70, 4);
    ctx.fill();

    // Box flaps
    ctx.strokeStyle = '#92400e';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(-50, 0);
    ctx.lineTo(50, 0);
    ctx.moveTo(0, -35);
    ctx.lineTo(0, 35);
    ctx.stroke();

    // Text "RECYCLABLE PAPER"
    ctx.fillStyle = '#78350f';
    ctx.font = 'bold 10px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('CARDBOARD BOX', 0, 4);

  } else if (materialId === 'plastic') {
    // 🥤 Render Plastic Bottle
    ctx.fillStyle = '#60a5fa';
    ctx.beginPath();
    ctx.ellipse(0, 5, 30, 55, 0, 0, Math.PI * 2);
    ctx.fill();

    // Bottle cap & neck
    ctx.fillStyle = '#1d4ed8';
    ctx.fillRect(-10, -60, 20, 15);
    ctx.fillStyle = '#2563eb';
    ctx.fillRect(-12, -68, 24, 8);

    // Reflections
    ctx.fillStyle = 'rgba(255, 255, 255, 0.4)';
    ctx.beginPath();
    ctx.ellipse(-10, 0, 6, 40, -0.2, 0, Math.PI * 2);
    ctx.fill();

    ctx.fillStyle = '#ffffff';
    ctx.font = 'bold 9px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('PET BOTTLE', 0, 8);

  } else if (materialId === 'metal') {
    // 🥫 Render Soda Can
    const grad = ctx.createLinearGradient(-35, 0, 35, 0);
    grad.addColorStop(0, '#9ca3af');
    grad.addColorStop(0.3, '#fef08a');
    grad.addColorStop(0.7, '#eab308');
    grad.addColorStop(1, '#854d0e');
    ctx.fillStyle = grad;
    ctx.beginPath();
    ctx.roundRect(-35, -55, 70, 110, 12);
    ctx.fill();

    // Can top tab
    ctx.fillStyle = '#d1d5db';
    ctx.beginPath();
    ctx.ellipse(0, -55, 30, 8, 0, 0, Math.PI * 2);
    ctx.fill();

    ctx.fillStyle = '#1e293b';
    ctx.font = 'bold 10px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('ALUMINUM CAN', 0, 4);

  } else if (materialId === 'glass') {
    // 🍾 Render Glass Bottle / Jar
    ctx.fillStyle = 'rgba(34, 197, 94, 0.6)';
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.8)';
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.roundRect(-32, -50, 64, 100, 16);
    ctx.fill();
    ctx.stroke();

    // Glass highlights
    ctx.fillStyle = 'rgba(255, 255, 255, 0.7)';
    ctx.fillRect(-24, -40, 8, 80);

    ctx.fillStyle = '#ffffff';
    ctx.font = 'bold 10px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('GLASS BOTTLE', 0, 4);
  }

  ctx.restore();

  // Tech overlay stats
  ctx.save();
  ctx.fillStyle = 'rgba(255, 255, 255, 0.7)';
  ctx.font = '10px monospace';
  ctx.fillText(`SORTIQ TEST FEED // SAMPLE: ${materialId.toUpperCase()} // STATUS: SCANNING`, 15, height - 15);
  ctx.restore();
}

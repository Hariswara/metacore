import {C} from './lib';
export const IconPeople: React.FC<{s?: number; c?: string}> = ({s = 64, c = C.cyan}) => (
  <svg width={s} height={s} viewBox="0 0 64 64" fill="none" stroke={c} strokeWidth={3.5} strokeLinecap="round">
    <circle cx={24} cy={20} r={8} /><path d="M8 52c0-10 7-16 16-16s16 6 16 16" />
    <circle cx={44} cy={22} r={6} /><path d="M40 36c8 0 16 5 16 14" />
  </svg>
);
export const IconHospital: React.FC<{s?: number; c?: string}> = ({s = 64, c = C.cyan}) => (
  <svg width={s} height={s} viewBox="0 0 64 64" fill="none" stroke={c} strokeWidth={3.5} strokeLinejoin="round">
    <rect x={8} y={8} width={48} height={48} rx={10} /><path d="M27 18h10v9h9v10h-9v9H27v-9h-9V27h9z" fill={c} fillOpacity={0.25} />
  </svg>
);
export const IconWater: React.FC<{s?: number; c?: string}> = ({s = 64, c = C.cyan}) => (
  <svg width={s} height={s} viewBox="0 0 64 64" fill="none" stroke={c} strokeWidth={3.5} strokeLinejoin="round">
    <path d="M32 6C24 20 14 30 14 40a18 18 0 0 0 36 0C50 30 40 20 32 6z" fill={c} fillOpacity={0.18} /><path d="M24 42a8 8 0 0 0 8 8" />
  </svg>
);

import {AbsoluteFill, Sequence} from 'remotion';
import tl from './timeline.json';
import {Overlay, Captions} from './lib';
import {S1} from './scenes/S1';
import {S2} from './scenes/S2';
import {S3} from './scenes/S3';
import {S4} from './scenes/S4';
import {S5} from './scenes/S5';
import {S6} from './scenes/S6';
import {S7} from './scenes/S7';
import {S8} from './scenes/S8';
import {S9} from './scenes/S9';
import {S10} from './scenes/S10';
import {S11} from './scenes/S11';
import {S12} from './scenes/S12';

const SC = [S1, S2, S3, S4, S5, S6, S7, S8, S9, S10, S11, S12];

export const Main: React.FC<{only?: number}> = ({only}) => (
  <AbsoluteFill style={{background: '#050B14'}}>
    {tl.scenes.map((s, i) => {
      const Comp = SC[i];
      if (only && only !== s.id) return null;
      return (
        <Sequence key={s.id} from={only ? 0 : s.from} durationInFrames={s.dur} name={`Scene ${s.id}`}>
          <Comp />
          <Captions scene={s.id} />
        </Sequence>
      );
    })}
    <Overlay />
  </AbsoluteFill>
);

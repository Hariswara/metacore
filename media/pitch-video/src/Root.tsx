import {Composition} from 'remotion';
import {Main} from './Main';
import tl from './timeline.json';
import '@fontsource/space-grotesk/400.css';
import '@fontsource/space-grotesk/500.css';
import '@fontsource/space-grotesk/700.css';
import '@fontsource/jetbrains-mono/400.css';
import '@fontsource/jetbrains-mono/700.css';

export const Root: React.FC = () => (
  <>
    <Composition id="Pitch" component={Main} durationInFrames={tl.total} fps={30} width={1920} height={1080} />
    {tl.scenes.map((s) => (
      <Composition key={s.id} id={`S${s.id}`} component={Main} durationInFrames={s.dur} fps={30} width={1920} height={1080} defaultProps={{only: s.id}} />
    ))}
  </>
);

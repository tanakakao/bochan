import { useEffect, useState, type ComponentType } from "react";

interface PersistentWorkbenchPageProps {
  active: boolean;
  cacheKey: string;
  pageId: string;
  Page: ComponentType;
}

/**
 * Keeps a visited page mounted while it is inactive so local UI state survives
 * ordinary navigation. Changing cacheKey invalidates that page and remounts it
 * only for the new dataset/result revision.
 */
export default function PersistentWorkbenchPage({
  active,
  cacheKey,
  pageId,
  Page
}: PersistentWorkbenchPageProps) {
  const [mountedKey, setMountedKey] = useState<string | null>(() => active ? cacheKey : null);

  useEffect(() => {
    if (active) {
      setMountedKey(cacheKey);
      return;
    }
    setMountedKey((current) => current !== null && current !== cacheKey ? null : current);
  }, [active, cacheKey]);

  useEffect(() => {
    if (!active) return;
    const frame = window.requestAnimationFrame(() => {
      window.dispatchEvent(new Event("resize"));
    });
    return () => window.cancelAnimationFrame(frame);
  }, [active, cacheKey]);

  const renderKey = active ? cacheKey : mountedKey;
  if (renderKey === null) return null;

  return (
    <div
      hidden={!active}
      aria-hidden={!active}
      data-workbench-page={pageId}
      data-page-cache-key={renderKey}
    >
      <Page key={renderKey} />
    </div>
  );
}

export const TUTORIAL_SAMPLE_DATASET_NAME = "bochan_tutorial_materials.csv";

const TEMPERATURES = [700, 750, 800, 850, 900];
const HOLD_TIMES = [20, 40, 60];
const ADDITIVE_RATIOS = [0.5, 1.5];

function calculateStrength(
  temperature: number,
  holdTime: number,
  additiveRatio: number,
  rowIndex: number
): number {
  const base = 122;
  const temperaturePenalty = 0.0016 * (temperature - 820) ** 2;
  const timePenalty = 0.018 * (holdTime - 45) ** 2;
  const additivePenalty = 6.5 * (additiveRatio - 1.2) ** 2;
  const deterministicVariation = 1.4 * Math.sin((rowIndex + 1) * 1.7);
  return Number((
    base - temperaturePenalty - timePenalty - additivePenalty + deterministicVariation
  ).toFixed(3));
}

export function createTutorialSampleFile(): File {
  const rows: Array<[number, number, number, number]> = [];
  let rowIndex = 0;

  TEMPERATURES.forEach((temperature) => {
    HOLD_TIMES.forEach((holdTime) => {
      ADDITIVE_RATIOS.forEach((additiveRatio) => {
        rows.push([
          temperature,
          holdTime,
          additiveRatio,
          calculateStrength(temperature, holdTime, additiveRatio, rowIndex)
        ]);
        rowIndex += 1;
      });
    });
  });

  const csv = [
    "temperature,hold_time,additive_ratio,strength",
    ...rows.map((row) => row.join(","))
  ].join("\r\n");

  return new File([`\uFEFF${csv}`], TUTORIAL_SAMPLE_DATASET_NAME, {
    type: "text/csv;charset=utf-8"
  });
}

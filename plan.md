# Swelling study outline

## Motivation:
- previous works have investigated effects of liquid redistribution within a vibration cycle with biphasic models
- in this study we want to explore the effects of liquid accumulation/swelling in the longer term where the accumulation of fluid does not vary greatly within a single vibration cycle
    - Such a swelling could be the long term result of liquid redistribution as in biphasic models or other bodily processes like healing that could cause local inflammation at the vocal folds
- The effects of this longer term swelling on vocal fold dynamics could have implication on the progressions of vocal disorders, like hyperfunction

### Questions to address
- Overall question: What is the effect of swelling on vocal fold dynamics? (collision pressure, SPL, frequency, measures of damage, others of importance for etiology of hyperfunction)
    - note that swelling has three distinct effects on vocal folds:
        - changes stiffness, due to function m(v)
        - changes mass due to swelled fluid entering
        - changes 'rest' geometry due to swelled fluid entering
    - elucidate the primary effect of swelling by considering variations of the three effects
        - consider what happens with a pure stiffness change?
        - consider what happens with a pure mass change?
        - consider what happens with a pure shape change?
        - have to consider how to compare the magnitude of change in stiffness rate as it's difficult to directly compare with the mass/shape changes

- Hypotheses
    - Swelling primarily changes dynamics by changing the rest geometry
        - This could be good or bad for voice usage depending on the glottal gap
        - If swelling changes the glottal gap to be more efficient*, then it could improve phonation
        - If swelling changes the glottal gap to be less efficient*, then it could worsen phonation

## Methodology

### Model
- Use a linearized isotropic model we have used in our previous works + add the swelling extension from Gou and Pence

### Parameter/Geometry variations
- Consider uniform swelling values in [1.0, 1.01, ..., 1.09, 1.10] confined to the SLP layer
    - Does literature have reasonable values for what the maximum swelling can be?
- Also consider the stiffness change rates of [0.8, 0.9, 1.0, 1.1, 1.2]
    - Does literature have reasonable values or justification how stiffness changes with swelling?
    - See the paper by Gou and Pence to see what they have considered
    - Yang et al (2017) provide estimated stress/strain curve values at different dehydration levels
        - at 0% loading / unloading = 23589.36 / 41374.50 = 32481.93
        - at 40% loading / unloading = 29810.85 / 44846.05 = 37328.45
        - if the vocal folds are 80% water by volume at 0% hydration (Phillips, etal 2009)
        - a 40% dehydration corresponds to a swelling of v=(0.2 + 0.8*0.6)/(0.2 + 0.8) = 0.68
        - the normalized stiffness change in the first dehydration step of Yang is 37328.45/32481.93=1.1492
        - Therefore m' = (normalized stiffness change-1)/(swelling - 1) = 0.1492 / (0.68-1) = -0.466
            - From the swelling model, it works out that the new modulus is given by
                - E_new = (1+m'*(v-1)) * (E_old)
                - Therefore
                    - m' = (E_new/E_old - 1)/(v-1)

- For layer properties, some example values from past studies are:
- Murray and Thomson (2012)

- Murray and Thomson mention in a brief review of past silicone models that Young's moduli for 2 layer silicone VFs range from
    - Ebody in [8 - 23] kPa
    - Ecover in [2 - 9] kPa

- Yang et al (2017) - Fully-coupled aeroelastic simulation with fluid compressibility — For application to vocal fold vibration
    - These authors set 3 different iostropic moduli cases
    - soft case - Ecover = 5 kPa, Ebody = 20kPa
    - stiff case - Ecover = 10 kPa, Ebody = 40kPa

- Link et al (2009) -
- These authors set a 3 layer model as
    - soft case - Ecover = 10 kPa, Ebody = 40 kPa, Eligament = 100kPa

- For this study, replicate the results by Hadwin and Peterson (2019) and use parameter variations for a M5 model with convergent medial angle
    - For the FastLab geometry, use 11.8/2/.6/45 kPa for body/ligament/cover/epithelium
    - For the convergent M5 geometry use cover properties of [2 - 5] kPa and body properties of [5 - 10] kPa

### Quantities to analyze
- Analyze qualitative signals of interest
    - glottal width/flow
- Analyze basic quantities of interest
    - Frequency, volume/amplitude
- Run transient simulations and report important quantities (frequency, dissipated energy dose, etc.)

## Results / discussion
- plot glottal width and flow VS swelling
    - note any consistent qualitative effects of increasing swelling on the waveforms
    - is it consistent across different body/cover moduli?
    - details:
        - segment waveforms on a per-period basis
        - align them against each other with cross correlation
        - overlay waveforms at each swelling state

- plot measures related to voice output (SPL, frequency) VS swelling
    - note if swelling decreases/increases SPL
    - is it consistent across different body/cover moduli?
        - if swelling decreases SPL, could imply it requires a compensatory effect that could exacerbate vocal disorders
    - details:
        - for each sweling condition, compute acoustic power (see calculations done in mask paper)
        - Plot trend of average acoustic power output vs level of swelling

- plot measures related to vocal fold damage (dissipated dose, max von Mises stress) VS swelling
    - details:
        - for each sweling condition, compute average dissipated dose and average von Mises stress
        - Plot trends of damage measures vs level of swelling
        - for these ones, will have to compute in a post-processing step

## Conclusion / hypotheses
- hypotheses:
    - increased swelling generally decreases vocal outputs (volume, maybe frequency)
        - if true, this would suggest that vocal swelling requires compensatory changes which could ultimately play a role in causing vocal disorders
    - increased swelling has minor effects on vocal damage
        - if true, this would suggest that swelling by itself doesn't cause damage to the VFs; compensatory changes from a decrease in voice outputs could increase damage, however

## Limitations
- 2D instead of 3D

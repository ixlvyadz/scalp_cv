// Text shown in the app. Plain, everyday words only.
// IMPORTANT: have the condition text reviewed against a reliable medical
// source (e.g. a dermatologist, AAD, Cleveland Clinic) before submitting.

export const APP = {
  // Fill these in.
  // Photos: put them at public/team/elvie.jpg and public/team/baberose.jpg.
  team: [
    { name: 'Elvie May P. Mara', title: 'Model Training Lead', role: 'Trained and tested the model', photo: 'team/elvie.jpg' },
    { name: 'Baberose I. Silmaro', title: 'App Design Lead', role: 'Designed and built the app', photo: 'team/baberose.jpg' },
  ],
  course: '[Course] · [Instructor]',
  // Set to a number (e.g. 86) once testing is finished. null shows a dash.
  accuracyOutOf100: null,
};

// key -> content. `aliases` lets the app match whatever spelling labels.json uses.
export const CONDITIONS = [
  {
    key: 'alopecia_areata',
    name: 'Alopecia areata',
    aliases: ['alopecia areata', 'alopecia'],
    short: 'Round, patchy hair loss with smooth bare spots.',
    subtitle: 'Patchy hair loss',
    facts: ['Not contagious', 'Hair can regrow'],
    what: 'A condition where the body’s own defence system attacks hair roots, causing hair to fall out in round or oval patches. The skin in the patch usually looks smooth and healthy.',
    signs: ['Round or oval bare patches', 'Smooth skin where hair fell out', 'Short broken hairs at the edge of a patch', 'Patches that appear quickly'],
    doctor: ['You notice a new bare patch', 'Patches are spreading or joining together', 'Eyebrows, eyelashes or nails are also affected'],
    good: 'Hair often grows back, and treatments can help. It is not caused by anything you did.',
  },
  {
    key: 'contact_dermatitis',
    name: 'Contact dermatitis',
    aliases: ['contact dermatitis'],
    short: 'Red, itchy skin after touching something that irritates it.',
    subtitle: 'A reaction to something on the skin',
    facts: ['Not contagious', 'Often has a trigger'],
    what: 'A skin reaction that happens when the scalp touches something that irritates it or causes an allergy, such as hair dye, shampoo or other hair products.',
    signs: ['Red, itchy or burning skin', 'Rash where a product touched the skin', 'Dry, cracked or scaly patches', 'Small blisters in some cases'],
    doctor: ['The rash is spreading or very painful', 'Your face or eyes swell', 'It does not get better after you stop using a product'],
    good: 'Finding and avoiding the trigger usually helps the skin settle down.',
  },
  {
    key: 'folliculitis',
    name: 'Folliculitis',
    aliases: ['folliculitis'],
    short: 'Small red bumps or pimples around hair roots.',
    subtitle: 'Inflamed hair roots',
    facts: ['Common', 'Often mild'],
    what: 'Inflammation of the hair roots (follicles), often caused by bacteria, irritation from shaving or tight hairstyles, or blocked follicles.',
    signs: ['Small red bumps around hairs', 'Pimples with a white tip', 'Itching or tenderness', 'Crusty sores in some cases'],
    doctor: ['Bumps are large, painful or spreading', 'You have a fever', 'It keeps coming back'],
    good: 'Mild cases often clear up with gentle care. A doctor can help if it does not improve.',
  },
  {
    key: 'head_lice',
    name: 'Head lice',
    aliases: ['head lice', 'lice', 'pediculosis'],
    short: 'Tiny insects in the hair that cause itching.',
    subtitle: 'Tiny insects in the hair',
    facts: ['Spreads by close contact', 'Treatable'],
    what: 'Small insects that live in the hair and feed on the scalp. They spread mainly through head-to-head contact. Having lice is not a sign of being unclean.',
    signs: ['Itching, especially behind the ears and at the neck', 'Tiny eggs (nits) stuck to hairs near the scalp', 'Small moving insects in the hair', 'Red bumps from scratching'],
    doctor: ['Treatments do not work', 'Scratching has caused sores or signs of infection', 'You are unsure whether it is lice'],
    good: 'Lice can be removed with the right treatment and careful combing.',
  },
  {
    key: 'lichen_planus',
    name: 'Lichen planus',
    aliases: ['lichen planus', 'lichen planopilaris'],
    short: 'Itchy purple-red patches that can damage hair roots.',
    subtitle: 'Inflammation that can affect hair roots',
    facts: ['Not contagious', 'See a doctor early'],
    what: 'An inflammatory skin condition. On the scalp it can affect hair roots and may lead to permanent hair loss if it is not treated.',
    signs: ['Itchy, purple-red bumps or patches', 'Redness and scaling around hairs', 'Patches of thinning hair', 'Tenderness or burning on the scalp'],
    doctor: ['You notice patchy hair loss with redness', 'Your scalp burns or hurts', 'Mouth, skin or nails also have changes'],
    good: 'Early treatment can help protect hair roots.',
  },
  {
    key: 'male_pattern_baldness',
    name: 'Male pattern baldness',
    aliases: ['male pattern baldness', 'androgenetic alopecia', 'male pattern hair loss'],
    short: 'Thinning at the temples and crown that runs in families.',
    subtitle: 'Gradual thinning that runs in families',
    facts: ['Very common', 'Not contagious'],
    what: 'Gradual hair thinning linked to genes and hormones. It usually starts at the temples or the crown and slowly progresses over years.',
    signs: ['Receding hairline at the temples', 'Thinning on the crown', 'Finer, shorter hairs', 'Slow change over months or years'],
    doctor: ['Hair loss is sudden or patchy', 'Your scalp is itchy, red or sore', 'You want to discuss treatment options'],
    good: 'It does not harm your overall health. Treatments can slow it if started early.',
  },
  {
    key: 'psoriasis',
    name: 'Psoriasis',
    aliases: ['psoriasis', 'scalp psoriasis'],
    short: 'Thick red patches covered with silvery scales.',
    subtitle: 'Thick, scaly patches',
    facts: ['Not contagious', 'Comes and goes'],
    what: 'A long-lasting condition where skin cells build up too quickly, forming thick, scaly patches. On the scalp it can extend past the hairline.',
    signs: ['Raised red patches', 'Silvery-white scales', 'Dry scalp that may crack or bleed', 'Itching or soreness'],
    doctor: ['Patches are spreading or painful', 'Your joints are stiff or swollen', 'Treatments are not helping'],
    good: 'There are many treatments that can keep it under control.',
  },
  {
    key: 'seborrheic_dermatitis',
    name: 'Seborrheic dermatitis',
    aliases: ['seborrheic dermatitis', 'seborrhoeic dermatitis', 'dandruff'],
    short: 'Greasy, yellowish flakes with redness.',
    subtitle: 'Often called dandruff when mild',
    facts: ['Common', 'Not contagious', 'Comes and goes'],
    what: 'A common skin condition that makes the scalp itchy, flaky and sometimes greasy. It often shows up where skin has many oil glands, like the scalp and face. It is not caused by poor hygiene.',
    resultWhat: 'Seborrheic dermatitis is a common skin condition that makes the scalp itchy, flaky and sometimes greasy. Mild cases are often called dandruff. It is not contagious, and it is not caused by poor hygiene.',
    signs: ['White or yellow flakes', 'Dry or greasy scaling on the scalp', 'Itchy, red skin', 'Thick, scaly patches in some cases'],
    resultSigns: ['White or yellow flakes', 'Itchy, red or greasy skin', 'Flare-ups that come and go'],
    lookalike: { key: 'psoriasis', text: 'Also red and scaly, but the scales are usually thicker and drier. A doctor can tell them apart.' },
    doctor: ['You are not sure what it is', 'It keeps coming back or does not improve', 'Scratching causes sores, bleeding or signs of infection'],
    good: 'It does not harm your overall health, and treatment can keep it under control.',
    source: 'Based on information from Cleveland Clinic and UAMS Health.',
  },
  {
    key: 'telogen_effluvium',
    name: 'Telogen effluvium',
    aliases: ['telogen effluvium'],
    short: 'Heavy shedding a few months after stress or illness.',
    subtitle: 'Shedding after stress or illness',
    facts: ['Not contagious', 'Often temporary'],
    what: 'Extra hair shedding that starts a few months after a shock to the body, such as illness, high fever, surgery, childbirth, stress or a big diet change.',
    signs: ['More hair than usual on your pillow or brush', 'Thinning all over, not in patches', 'Shedding that started a few months after a stressful event', 'Scalp skin looks normal'],
    doctor: ['Shedding lasts more than six months', 'You also feel tired or unwell', 'You see bare patches'],
    good: 'Hair usually grows back on its own once the cause has passed.',
  },
  {
    key: 'tinea_capitis',
    name: 'Tinea capitis',
    aliases: ['tinea capitis', 'scalp ringworm', 'ringworm'],
    short: 'A fungal infection with scaly patches and broken hairs.',
    subtitle: 'A fungal scalp infection',
    facts: ['Can spread', 'Needs treatment'],
    what: 'A fungal infection of the scalp and hair, most common in children. It can spread through close contact or shared combs, hats and pillows.',
    signs: ['Scaly, itchy patches', 'Broken hairs or black dots on the scalp', 'Patches of hair loss', 'Swollen, tender areas in some cases'],
    doctor: ['You think it may be a fungal infection', 'There is pus, swelling or pain', 'Others in your home have similar patches'],
    good: 'It usually clears with medicine from a doctor. Avoid sharing combs and hats meanwhile.',
  },
];

const norm = (s) => String(s).toLowerCase().replace(/[^a-z]/g, '');
const BY_KEY = Object.fromEntries(CONDITIONS.map((c) => [c.key, c]));

export function conditionByKey(key) {
  return BY_KEY[key];
}

// Match a label from labels.json (any spelling) to a condition.
export function conditionForLabel(label) {
  const n = norm(label);
  for (const c of CONDITIONS) {
    if (norm(c.key) === n || norm(c.name) === n) return c;
    if (c.aliases.some((a) => norm(a) === n)) return c;
  }
  for (const c of CONDITIONS) {
    if (c.aliases.some((a) => n.includes(norm(a)) || norm(a).includes(n))) return c;
  }
  // Unknown label: show it as-is with neutral text.
  const name = String(label).replace(/[_-]+/g, ' ').replace(/\b\w/g, (m) => m.toUpperCase());
  return { key: 'unknown_' + n, name, aliases: [], short: '', subtitle: '', facts: [], what: '', signs: [], doctor: ['You are not sure what it is'], good: '' };
}

// Optional photos: drop JPGs named <key>.jpg into public/samples/ (only images you have permission to use).
export const samplePhoto = (key) => `samples/${key}.jpg`;

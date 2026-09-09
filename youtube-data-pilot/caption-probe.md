# Caption probe — YouTube pilot

`scripts/probe_youtube_captions.py` over 1 supplied URLs. No audio downloaded. **1 accepted, 0 rejected.**

Every number below is a snapshot of one fetch, not a property of the video: YouTube's caption endpoint serves different recognition revisions for the same key from one extraction to the next. `punctuation per 100 words` says which one was captured — around 4 for the plain revision, around 14 for the punctuated one, which is also the better transcription. See this script's docstring.

Reference points for the two rates that decide review effort, measured 2026-08-11 over every `manifest.*.jsonl` in the existing corpora: `dataset/paid-dataset-v2` (synthetic, 74,542 words) has a **7.12%** lexical-particle rate and 7.23% non-Vietnamese-shaped words; `dataset/real-meetings-bench` (real speech, human-edited, 7,168 words) has **2.87%** and 8.75%. The bench figure is itself a floor — its reference is post-edited PhoWhisper-small output (PROJECT_CORE.md §4).

| Video | Duration | Verdict | Particle rate | EN words/min | wpm first→last |
|---|---:|---|---:|---:|---|
| `NZiW4QH83CI` | 60.2 min | accept | 0.64% | 6.9 | 106→165 |

## Livestream về RPA và AI. Tại sao các công ty lớn chi triệu đô mua và áp dụng RPA

- `video_id`: `NZiW4QH83CI` · https://www.youtube.com/watch?v=NZiW4QH83CI
- duration: 60.2 min
- **ACCEPT** — accepted
- `automatic_captions`: 157 keys, recognition original ['vi-orig']
- `subtitles`: 1 keys ['live_chat']

| Measurement | Value |
|---|---|
| json3 events / segs | 2618 / 13158 |
| segs with `tOffsetMs` | 10517 |
| words | 11857 |
| word entries covering >1 token | 7 |
| lexical-particle rate | 0.64% |
| punctuation per 100 words | 0.0 |
| digit-bearing words | 0.61% |
| non-Vietnamese-shaped words | 3.50% (6.9 / min) |
| bracketed sound labels | {'[âm nhạc]': 5} |
| words per fetch attempt | [11850] |

wpm per bucket: 0min=106, 5min=218, 10min=235, 15min=222, 20min=220, 25min=198, 30min=204, 35min=194, 40min=211, 45min=197, 50min=201, 55min=165

non-Vietnamese-shaped examples: api, robot, ok, boss, email, web, apa, fpt, iii, ipad, file, lôgic, code, excel, livestream

draft opening: `ừ ừ [âm nhạc] ừ ừ [âm nhạc] [âm nhạc] ừ ừ [âm nhạc] cho sự nghiệp của bạn đang có quá nhiều những công việc phải là vì lặp lại mỗi ngày nhân viên của bạn thì quá tải và nhàm chán trong khi bạn thì vẫn đang đều đạt tiêu tốn hàng núi tiền năng suất không tăng và sai sót thì thường xuyên xảy ra hiệu Đâ`

<details><summary>raw <code>automatic_captions</code> keys and <code>name</code> fields</summary>

```
aa ab af ak am ar as ay az ba be bg bho bn bo br bs ca ceb co crs cs cy da de dv dz ee el en eo es et eu fa fi fil fj fo fr fy ga gaa gd gl gn gu gv ha haw hi hmn hr ht hu hy id ig is it iu iw ja jv ka kha kk kl km kn ko kri ku ky la lb lg ln lo lt lua luo lv mfe mg mi mk ml mn mr ms mt my ne new nl no nso ny oc om or os pa pam pl ps pt pt-PT qu rn ro ru rw sa sd sg si sk sl sm sn so sq sr ss st su sv sw ta te tg th ti tk tn to tr ts tt tum ug uk ur uz ve vi vi-orig war wo xh yi yo zh-Hans zh-Hant zu

aa = 'Afar'
ab = 'Abkhazian'
af = 'Afrikaans'
ak = 'Akan'
am = 'Amharic'
ar = 'Arabic'
as = 'Assamese'
ay = 'Aymara'
az = 'Azerbaijani'
ba = 'Bashkir'
be = 'Belarusian'
bg = 'Bulgarian'
bho = 'Bhojpuri'
bn = 'Bangla'
bo = 'Tibetan'
br = 'Breton'
bs = 'Bosnian'
ca = 'Catalan'
ceb = 'Cebuano'
co = 'Corsican'
crs = 'Seselwa Creole French'
cs = 'Czech'
cy = 'Welsh'
da = 'Danish'
de = 'German'
dv = 'Divehi'
dz = 'Dzongkha'
ee = 'Ewe'
el = 'Greek'
en = 'English'
eo = 'Esperanto'
es = 'Spanish'
et = 'Estonian'
eu = 'Basque'
fa = 'Persian'
fi = 'Finnish'
fil = 'Filipino'
fj = 'Fijian'
fo = 'Faroese'
fr = 'French'
fy = 'Western Frisian'
ga = 'Irish'
gaa = 'Ga'
gd = 'Scottish Gaelic'
gl = 'Galician'
gn = 'Guarani'
gu = 'Gujarati'
gv = 'Manx'
ha = 'Hausa'
haw = 'Hawaiian'
hi = 'Hindi'
hmn = 'Hmong'
hr = 'Croatian'
ht = 'Haitian Creole'
hu = 'Hungarian'
hy = 'Armenian'
id = 'Indonesian'
ig = 'Igbo'
is = 'Icelandic'
it = 'Italian'
iu = 'Inuktitut'
iw = 'Hebrew'
ja = 'Japanese'
jv = 'Javanese'
ka = 'Georgian'
kha = 'Khasi'
kk = 'Kazakh'
kl = 'Kalaallisut'
km = 'Khmer'
kn = 'Kannada'
ko = 'Korean'
kri = 'Krio'
ku = 'Kurdish'
ky = 'Kyrgyz'
la = 'Latin'
lb = 'Luxembourgish'
lg = 'Ganda'
ln = 'Lingala'
lo = 'Lao'
lt = 'Lithuanian'
lua = 'Luba-Lulua'
luo = 'Luo'
lv = 'Latvian'
mfe = 'Morisyen'
mg = 'Malagasy'
mi = 'Māori'
mk = 'Macedonian'
ml = 'Malayalam'
mn = 'Mongolian'
mr = 'Marathi'
ms = 'Malay'
mt = 'Maltese'
my = 'Burmese'
ne = 'Nepali'
new = 'Newari'
nl = 'Dutch'
no = 'Norwegian'
nso = 'Northern Sotho'
ny = 'Nyanja'
oc = 'Occitan'
om = 'Oromo'
or = 'Odia'
os = 'Ossetic'
pa = 'Punjabi'
pam = 'Pampanga'
pl = 'Polish'
ps = 'Pashto'
pt = 'Portuguese'
pt-PT = 'Portuguese (Portugal)'
qu = 'Quechua'
rn = 'Rundi'
ro = 'Romanian'
ru = 'Russian'
rw = 'Kinyarwanda'
sa = 'Sanskrit'
sd = 'Sindhi'
sg = 'Sango'
si = 'Sinhala'
sk = 'Slovak'
sl = 'Slovenian'
sm = 'Samoan'
sn = 'Shona'
so = 'Somali'
sq = 'Albanian'
sr = 'Serbian'
ss = 'Swati'
st = 'Southern Sotho'
su = 'Sundanese'
sv = 'Swedish'
sw = 'Swahili'
ta = 'Tamil'
te = 'Telugu'
tg = 'Tajik'
th = 'Thai'
ti = 'Tigrinya'
tk = 'Turkmen'
tn = 'Tswana'
to = 'Tongan'
tr = 'Turkish'
ts = 'Tsonga'
tt = 'Tatar'
tum = 'Tumbuka'
ug = 'Uyghur'
uk = 'Ukrainian'
ur = 'Urdu'
uz = 'Uzbek'
ve = 'Venda'
vi = 'Vietnamese'
vi-orig = 'Vietnamese (Original)'
war = 'Waray'
wo = 'Wolof'
xh = 'Xhosa'
yi = 'Yiddish'
yo = 'Yoruba'
zh-Hans = 'Chinese (Simplified)'
zh-Hant = 'Chinese (Traditional)'
zu = 'Zulu'
```
</details>

<details><summary>raw <code>subtitles</code> <code>name</code> fields</summary>

```
live_chat = None
```
</details>

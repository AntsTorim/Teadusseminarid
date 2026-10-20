# Lineaaralgebrast transformerini: 12 seminari

## Eesmärk

Lugeda semestri lõpus koos läbi artikkel "Attention is All You Need" (Vaswani jt, 2017) ja mõista, mis selles kirjas on. See artikkel kirjeldab transformeri arhitektuuri, millel põhinevad kõik tänased suured keelemudelid (GPT, Claude, Llama). Artikkel ise on lühike, aga eeldab lineaaralgebra vaba valdamist – seetõttu algab kava sealt.

## Vorm

60 minutit nädalas, eeltöö osalejale maksimaalselt 30 minutit. Igal kohtumisel on määratud ettekandja (või paar), kelle ettevalmistus on 2–3 tundi.

## Läbiv idee

Keelemudel on lõpuks hunnik maatrikskorrutisi. Iga kohtumine lisab ühe tüki, mida viimasel korral artiklis ära tuntakse.

---

# Plokk I: geomeetria (kohtumised 1–4)

## 1. Maatriks kui teisendus

**Seos LLM-iga.** Närvivõrgu kiht on maatriks. Kui laseme andmed läbi kihi, korrutame need maatriksiga. Kogu ülejäänud kursus on selle ühe tehte kordamine erinevates kohtades, seega tasub sellest alustada.

**Eeltöö (25 min).** 3Blue1Brown, "Essence of Linear Algebra", osad 1–3.

**Kohtumisel.**

Maatriks kui ruumi venitamine ja pööramine. Joonistame numpy's ühikruudu, korrutame maatriksiga, joonistame tulemuse. Iga osaleja teeb ühe teisenduse ja teised arvavad ära, milline maatriks selle taga oli.

Viimased 10 minutit:

*Tähistuskokkulepe.* Kirjutame kokku ja paneme märkmete repositooriumisse:

- **y = Wx** – põhivalem. W on kaalude maatriks (ingl *weights*, need on need arvud, mida mudel treeningul õpib), x on sisendvektor, y on väljund.
- **X** (suur täht) on andmemaatriks, kus read on vaatlused ja veerud tunnused.
- **d** tähistab mõõtmete arvu.

Sellest kokkuleppest peame kinni kuni lõpuni – kohtumisel 10 tulevad W_Q, W_K ja W_V, mis on lihtsalt kolm sellist W-d, ja siis pole midagi ümber õppida.

*Demo.* PyTorchi `nn.Linear(2, 2, bias=False)` sama ruudu peal:

```python
import torch
import torch.nn as nn

layer = nn.Linear(2, 2, bias=False)
with torch.no_grad():
    layer.weight.copy_(torch.tensor([[2.0, 1.0],
                                     [0.0, 1.0]]))

square = torch.tensor([[0.0, 0.0],
                       [1.0, 0.0],
                       [1.0, 1.0],
                       [0.0, 1.0]])

print(layer(square))    # venitatud rööpküliku nurgad
```

Näitame, et `layer(x)` annab täpselt sama tulemuse kui `layer.weight @ x`. Ütleme kohe välja, et tahvlil on vektorid veerud ja koodis read – sama tehe, teistpidi kokkulepe:

```python
torch.allclose(layer(X), X @ layer.weight.T)   # True
```

---

## 2. Maatrikskorrutis ja dimensioonid

**Seos LLM-iga.** Kaks asja. Esiteks: transformeris tehakse kümneid maatrikskorrutisi järjest ja artikli lugemine tähendab suuresti mõõtmete jälgimist. Teiseks: siin selgub, miks võrkudes on aktivatsioonifunktsioone üldse vaja.

**Eeltöö (20 min).** 3B1B osad 4–5.

**Kohtumisel.**

Korrutis kui teisenduste kompositsioon: kõigepealt üks teisendus, siis teine, ja tulemus on jälle üks maatriks.

Sellest järeldub midagi olulist: kui laseme andmed läbi kahe kihi järjest ilma vahepeal midagi tegemata, on tulemus sama, mis ühe kihiga. Sügavus ei anna iseenesest mitte midagi. Just seetõttu pannakse kihtide vahele **aktivatsioonifunktsioon** (mittelineaarne funktsioon, näiteks ReLU, mis lõikab negatiivsed väärtused nulliks) – see lõhub kompositsiooni ja teeb sügavuse mõttekaks.

Ülejäänud aeg on trenn. Kirjutame tahvlile mõõtmeid ja küsime, mis tuleb:

```
(batch, seq, d_model) @ (d_model, d_k) → ?
```

Siin `batch` on paralleelselt töödeldavate näidete arv, `seq` on teksti pikkus tokenites, `d_model` on mudeli sisemine mõõde. Neid tähiseid kasutatakse artiklis ja Karpathy koodis, seega tasub nendega juba harjuda, isegi kui nende sisu on veel udune.

---

## 3. Skalaarkorrutis, sarnasus ja projektsioon

**Seos LLM-iga.** See on kursuse pöördepunkt. Attention – transformeri keskne mehhanism – otsustab, milline sõna tekstis millisele tähelepanu pöörab. See otsus on lõpuks skalaarkorrutiste tabel. Kui skalaarkorrutis kui sarnasusmõõt on selge, on kohtumine 10 poole lihtsam.

**Eeltöö (15 min).** 3B1B osa 9.

**Kohtumisel.**

Esimene pool: skalaarkorrutis mõõdab, kui palju kaks vektorit "ühes suunas" osutavad. **Kosinussarnasus** on sama asi, normeeritud vektori pikkuse mõjust puhtaks. Teeme käsitsi paar kolmemõõtmelist sõnavektorit ja arvutame nendevahelised sarnasused.

Teine pool: **vähimruutude regressioon kui projektsioon.** Kirjutame y = Xβ, kus X on andmemaatriks ja β on otsitavad kordajad. Lahendus on see, kus jääk (viga) on veeruruumiga risti – sellest tulebki normaalvõrrand. Geomeetriliselt: projitseerime y selle ruumi peale, mida andmed katta suudavad.

Siin tuleb hästi välja, miks tähistus loeb. Kohtumisel 1 oli W see, mida otsime. Siin on X ette antud ja β see, mida otsime. Sama valemikuju, vastupidine tähendus.

> **Ajakasutus.** See on kava kõige tihedam kohtumine. Kui aeg kipub üle minema, lühenda kosinussarnasuse demot, mitte regressiooni.

---

## 4. Baas ja baasivahetus

**Seos LLM-iga.** Attention töötab nii, et sama sisendvektor projitseeritakse kolme erinevasse ruumi ja neid kasutatakse eri otstarbeks. Selle mõistmiseks peab olema selge, et sama informatsiooni saab esitada erinevates koordinaadistikes ja et projektsioon alamruumi on tavaline õpitav operatsioon.

**Eeltöö (20 min).** 3B1B osa 13.

**Kohtumisel.**

Sama vektor eri baasides. Baasivahetuse maatriks. Projektsioon madalamamõõtmelisse alamruumi ja see, mis seejuures kaotsi läheb.

Viimased 5 minutit, etteruttamiseks: transformeris korrutatakse iga sõna vektorit kolme erineva õpitud maatriksiga, mida tähistatakse **W_Q**, **W_K** ja **W_V**. Need on tavalised kaalumaatriksid, täpselt nagu kohtumise 1 W. Indeksid tulevad sõnadest *query*, *key* ja *value* – nende sisuline roll tuleb kohtumisel 10. Praegu piisab teadmisest, et tegu on kolme projektsiooniga samast sisendist.

---

# Plokk II: maatriksitest õpitud esitusteni (kohtumised 5–8)

## 5. SVD ja PCA

**Seos LLM-iga.** Keelemudel esitab sõnu vektoritena mitmesajamõõtmelises ruumis. Miks see üldse töötab? Vastus on, et andmed elavad tegelikult palju madalamas dimensioonis, kui nad välja näevad, ja seda madalamat ruumi saab leida. PCA on selle idee kõige puhtam näide ja ajalooliselt esimene viis, kuidas sõnu vektoriteks tehti. Võhandu "sibula koorimine".

**Eeltöö (25 min).** 3B1B osa 14 (omaväärtused ja omavektorid) + Steve Bruntoni "SVD Overview". Omaväärtusi kohtumisel enam üle ei korrata, need peavad eeltööst tulema.

**Kohtumisel kolm osa.**

### ~20 min. SVD

Iga maatriksi saab lahutada kujule X = UΣVᵀ, mida saab lugeda kui pööre → venitus → pööre. Σ on diagonaalmaatriks ja tema arvud (singulaarväärtused) ütlevad, kui palju igas suunas venitatakse.

Kirjuta tahvlile ka summakuju:

```
X = σ₁u₁v₁ᵀ + σ₂u₂v₂ᵀ + σ₃u₃v₃ᵀ + …
```

Iga liidetav on astaku-1 kiht ja σ ütleb, kui paks see kiht on. Lase rühmal järjest liikmeid juurde panna ja vaadata, kuidas algne maatriks tasapisi taastub. Dimensiooni vähendamine tähendabki välimiste, õhemate kihtide äraviskamist – maatriksi koorimist.

### ~25 min. PCA

**Peakomponentanalüüs** otsib andmetes need suunad, mille sihis on kõige rohkem varieeruvust. Retsept: lahuta X-ist veergude keskmised (tsentreerimine), tee SVD, ja V veerud ongi peakomponendid. Σ ruudud ütlevad, kui palju iga komponent dispersioonist seletab.

```python
Xc = X - X.mean(axis=0)
U, S, Vt = np.linalg.svd(Xc, full_matrices=False)

pca = PCA().fit(X)
# Vt ja pca.components_ langevad kokku (märgi täpsuseni)
```

Tähelepanuväärne on see, mida **ei** tehta: kovariatsioonimaatriksit ei arvutata. Sama tulemus tuleb otse andmemaatriksist ja numbriliselt stabiilsemalt. Nii teeb ka `sklearn`.

Hoiatus ette: komponendi märk on määramata, seega kui `sklearn` annab vastupidise suunaga vektori, ei ole midagi katki.

### ~15 min. Latentne ruum

Vähendame dimensiooni ja vaatame, mis alles jääb. Kui kasutame termin-dokument maatriksit (read = sõnad, veerud = dokumendid, väärtused = esinemissagedused), saame **latentse semantilise analüüsi (LSA)** – 1990. aastate meetod, mis leidis dokumentidest teemad ilma neid ette andmata. See on otsene eellane sellele, mis tuleb järgmisel korral.

> **Valikuline lisa (10 min): monotoonsed süsteemid.**
> Kohalik vastukaal: Võhandu **mõjuskaala** on odav (O(MN)) viis andmetabeli ridu ja veerge korrastada nii, et struktuur muutub palja silmaga nähtavaks – ilma igasuguse lahutuseta. Mullati tuumaeraldusalgoritmi keskne protseduur kannab nime KIHT: kui tuum on leitud, elimineeritakse selle elemendid tabelist ja allesjäänu on uus lähteandmestik. Kiht kihi haaval, kuni midagi järele ei jää.
> Allikas: Võhandu, Kuusik, Lind, *Monotoonsed süsteemid ja nende rakendused* (2018), TalTechi digikogus vabalt saadaval. Näitetabel on kuuerealine ja mahub tahvlile. Algoritmide samm-sammulise töö videod on YouTube'is.

---

## 6. Embeddingud

**Seos LLM-iga.** Mudel ei näe sõnu vaid arve. **Embedding** on sõna (täpsemalt tokeni) esitus vektorina, ja kogu sõnastiku embeddingud koos moodustavad ühe suure maatriksi. See on transformeri esimene kiht ja seega esimene päris LLM-i komponent, mille lahti võtame.

**Eeltöö (20 min).** 3Blue1Brown, "But what is a GPT?", esimene kolmandik (kuni attentionini).

**Kohtumisel.**

Embedding-tabel on maatriks, mille ridu valitakse tokeni numbri järgi. Rohkem seal ei ole. **Token** on tekstitükk, millega mudel opereerib – tavaliselt sõna või sõnaosa, mitte täht.

Laadime alla valmis embeddingud, otsime lähimaid naabreid, teeme klassikalise `king - man + woman ≈ queen` katse. Vaatame ka, kus see katse ei tööta, sest neid kohti on rohkem kui tutvustustes näidatakse.

Siis piirang, mis seob eelmise korraga: PCA leiab lineaarse alamruumi ja maksimeerib dispersiooni. Embeddingud õpitakse ülesande kaudu (ennusta järgmine sõna) ja on paremad, sest ülesanne ütleb, mis on oluline. Kaar on LSA → word2vec → kontekstipõhised esitused, ja viimane on täpselt see, mida attention teeb: sama sõna saab erineva vektori sõltuvalt sellest, mis teda ümbritseb.

---

## 7. Gradient ja ahelreegel

**Seos LLM-iga.** Seni oleme kaalud käsitsi paika pannud. Treenimine tähendab, et need leitakse andmetest, ja meetodit selleks nimetatakse **tagasilevi**ks (*backpropagation*). See on ainus koht kursusel, kus on vaja tuletisi.

**Eeltöö (25 min).** 3Blue1Brown, "Backpropagation calculus".

**Kohtumisel.**

Kahekihilise võrgu gradiendid käsitsi läbi arvutatud. Ahelreegel on kogu mehhanism: viga jookseb väljundist tagasi ja iga kaal saab teada, kui palju tema seda viga mõjutas.

Kontrollime numbriliselt – muudame üht kaalu tibake ja vaatame, kas kadu muutub nii palju, kui gradient ennustas. See on standardne võte, mille kirjeldamine võtab minuti ja mis säästab hiljem tunde silumist.

Kohtumise 1 W-d, mis olid käsitsi paika pandud, hakkavad nüüd andmetest tulema.

*Kui rühm valdab seda juba, saab kohtumised 7 ja 8 kokku tõmmata.*

---

## 8. Softmax, kadufunktsioon ja treeningtsükkel

**Seos LLM-iga.** **Softmax** teeb suvalisest arvude komplektist tõenäosusjaotuse (kõik positiivsed, summa üks). Keelemudelis kasutatakse teda kaks korda: väljundis, et saada järgmise tokeni tõenäosused, ja attentioni sees, et saada tähelepanukaalud. Teine kasutus on see, mis meid kohtumisel 10 huvitab.

**Eeltöö (25 min).** Karpathy "Neural Networks: Zero to Hero", micrograd, määratud ajatemplid.

**Kohtumisel.**

Softmaxi valem ja tema käitumine: mis juhtub, kui sisendarvud on väga suured (üks võidab kõik, jaotus muutub teravaks) ja kui väga väikesed (jaotus muutub ühtlaseks). Seda tasub tähele panna, sest kohtumisel 11 on see täpselt see probleem, mille pärast √d jagamine sisse toodi.

**Kadufunktsioon** (*loss*) mõõdab, kui vale mudeli ennustus oli. Treeningtsükkel: ennusta → arvuta kadu → arvuta gradiendid → uuenda kaale → korda.

Treenime väikese mudeli päriselt ära ja vaatame kadu langemas.

---

# Plokk III: attention ja transformer (kohtumised 9–12)

## 9. Miks on konteksti vaja

**Seos LLM-iga.** Enne lahenduse vaatamist on vaja aru saada, mis probleemi see lahendab. Muidu on attention lihtsalt suvaline valem, mille keegi välja mõtles.

**Eeltöö (25 min).** Karpathy, "Let's build GPT", algusest kuni bigrammi mudeli valmimiseni. Video on kaks tundi pikk, seega konkreetsed ajatemplid tuleb ette anda.

**Kohtumisel.**

**Bigrammi mudel** ennustab järgmist tähte ainult eelmise tähe põhjal. Treenime selle ära ja laseme teksti genereerida. Tulemus on äratuntavalt keelesarnane prügi.

Diagnoos: mudel ei näe minevikku. "Pank" tähendab eri asja lauses raha kohta ja lauses jõe kohta, aga bigrammi mudelil pole mingit võimalust seda vahet teha. Vaja on mehhanismi, mis laseb igal positsioonil vaadata teisi positsioone ja võtta neilt informatsiooni.

Sellega on probleem püstitatud ja järgmine kohtumine on vastus.

---

## 10. Self-attention

**Seos LLM-iga.** See on artikli süda ja kogu kursuse siht.

**Eeltöö (25 min).** Karpathy sama video, self-attentioni lõik.

**Kohtumisel.** Arvutame attentioni maatriksi käsitsi tahvlil. Neli tokenit, kahemõõtmelised vektorid, arvud sellised, et peast saab arvutada.

Käik on järgmine. Iga tokeni vektorist x teeme kolm uut vektorit:

- **q = W_Q x** (*query*, päring) – mida see token otsib
- **k = W_K x** (*key*, võti) – mida see token pakub
- **v = W_V x** (*value*, väärtus) – mida see token edasi annab, kui teda valitakse

W_Q, W_K ja W_V on kolm õpitud kaalumaatriksit, täpselt sama tüüpi objektid nagu kohtumise 1 W. Nad on erinevad, sest neid treenitakse erinevatele asjadele.

Siis: iga päringu ja iga võtme vahel arvutame skalaarkorrutise. Saame tabeli, kus igas lahtris on "kui hästi token i päring sobib token j võtmega". Selle tabeli iga rea peale rakendame softmaxi ja saame kaalud, mis summeeruvad üheks. Lõpuks võtame väärtusvektoritest nende kaaludega kaalutud keskmise.

**See ongi kogu attention. Kaalutud keskmine, kus kaalud tulevad sarnasusest.**

Siin maandub korraga kolm varasemat kohtumist: skalaarkorrutis sarnasusena (3), projektsioon eri ruumidesse (4), softmax jaotuse tegijana (8). Tasub see välja öelda – see on hetk, mille pärast kogu eelnev tehtud sai.

**Viimased ~7 min: attention kui graaf**

Pane tähele, mis see tahvlil olev tabel õieti on. Pärast softmaxi summeerub iga rida üheks – see on realt stohhastiline maatriks. Ehk teisisõnu: kaalustatud suunatud graaf, kus tokenid on tipud ja kaal ütleb, kui palju token i võtab infot tokenilt j.
Sama tüüpi objekt on PageRanki üleminekumaatriks. Seal on tipud veebilehed, kaal 1 / väljundaste, ja tähtsus leitakse dominantse omavektorina astmemeetodiga: r ← Mr, kuni koondub. Google ehitati selle peale – astmemeetod kohtumise 5 eeltööst ei ole abstraktsioon.
Sarnasus ulatub kaugemale kui maatriksi kuju. Transformeri kihid on sama korrutamise kordamine, nagu astmemeetodi iteratsioonid: iga kiht laseb infol ühe sammu võrra kaugemale liikuda.

Ja siis erinevus, mis on kogu asja iva:
PageRankil on graaf ette antud ja kaalud fikseeritud. Attentionil arvutatakse graaf sisust, iga sisendi jaoks uuesti.
Sama lause teistpidi: attention on PageRank, mille graafi mudel ise iga kord välja mõtleb. Kui rühm mäletab kohtumisest 10 ainult ühte lauset, siis võiks see olla see.

---

## 11. Multi-head, skaleerimine, maskeerimine, positsioonid

**Seos LLM-iga.** Kohtumisel 10 tehtud attention on põhiidee, aga päris mudelis on tema ümber neli detaili, ilma milleta ta ei tööta. Kõik neli on artiklis ja kõik neli näevad esmapilgul välja nagu suvalised nipid.

**Eeltöö (20 min).** 3Blue1Brown, "Attention in transformers".

**Kohtumisel.**

*Skaleerimine √d-ga.* Kui vektorid on pikad, tulevad skalaarkorrutised suured, ja suurte arvude peal muutub softmax teravaks – üks token võidab kõik, gradiendid kaovad. Jagamine ruutjuurega mõõtmest hoiab arvud mõistlikus vahemikus. Kontrollime numbriliselt: arvutame sama asja skaleerimisega ja ilma, vaatame jaotust.

*Mitu pead.* Üks attentioni komplekt õpib ühte tüüpi seost. Mitu paralleelset komplekti (igaühel oma W_Q, W_K, W_V) õpivad erinevaid: üks süntaktilist sõltuvust, teine kaugemat viidet. Tulemused liidetakse kokku. Seetõttu **multi-head attention**.

*Maskeerimine.* Teksti genereerival mudelil ei tohi lasta tulevikku vaadata, muidu on ülesanne triviaalne. Tehniliselt: enne softmaxi pannakse tabeli ülemine kolmnurk miinus lõpmatuseks.

*Positsioonikodeering.* Attention ise on järjekorra suhtes pime – ta näeb tokenite kogumit, mitte järjendit. Positsiooniinfo tuleb eraldi sisse lisada.

---

## 12. Artikkel ise

**Eeltöö (30 min).** Vaswani jt, "Attention is All You Need", peatükid 3.1–3.5. Ülejäänud artikkel pole vajalik.

**Kohtumisel.**

Võtame lahti joonise 1 ja seome iga ploki eelmiste kohtumistega:

- Valem 1 = kohtumine 10 + kohtumise 11 skaleerimine
- Peatükk 3.2.2 = mitu pead
- Peatükk 3.3 (feed-forward) = kohtumise 1 y = Wx kahekordselt, aktivatsioon vahel
- Peatükk 3.5 = positsioonikodeering

Arutelu kaks poolt: mida artikkel halvasti või puudulikult seletab, ja mis on vahepeal muutunud. Tänastes mudelites on normaliseerimine mujal (pre-LN), positsioonikodeering teistsugune (RoPE), normaliseerimisviis lihtsam (RMSNorm), attentioni peade struktuur ökonoomsem (GQA). Artikli põhiskelett on aga sama, mis 2017. aastal.

Lõpetuseks üks laiem märkus: LSA, PCA ja monotoonsete süsteemide tuumad tegelevad kõik sama küsimusega – kus andmetes on struktuur – ja kolm eri koolkonda jõudsid selleni kolme eri teed. Attention on selles reas neljas.

---

# Praktilised märkused korraldajale

**Videote tükeldamine.** Karpathy videod on 1–2 tundi. Nendest tuleb ette ära valida konkreetsed ajatemplid, muidu eeltöö eelarvest ei jää midagi järele. Selle töö peab keegi enne semestri algust ära tegema.

**Eeltöö eelarve kehtib osalejale.** Ettekandja kulutab 2–3 tundi. Ütle see algusest välja, muidu tuleb keegi kohale mõttega, et 25 minutit videot on ettevalmistus.

**Ettekandja.** Määratud ettekandja funktsioon ei ole "keegi räägib", vaid "keegi on sunnitud materjali enne läbi töötama". Toimivad alternatiivid: paar (kaks valmistavad, üks räägib); rotatsioon ilma ettekandeta (määratud inimene toob töötava märkmiku ja kaks küsimust). Ei toimi: "kõik loevad ja siis arutame" – ilma nimelise vastutuseta muutub kohtumine kolmandaks korraks juhendaja loenguks.

**Järjekord kuu ette paika**, mitte nädal ette. Lühike etteteatamine on kõige sagedasem põhjus, miks keegi valmistumata tuleb. Raskemad korrad (9–12) pane nendele, kes tulevad toime; algusesse need, kelle jaoks ettekandmine ise on uus.

**Kus osalus kaob.** Tavaliselt kohtumiste 5–7 paiku, kus tundub, et LLM-idest ollakse kaugel. Vastumürk on iga kord ühe lausega öelda, kuhu see tükk hiljem läheb – seetõttu on igal kohtumisel eespool "seos LLM-iga" kirjas.

**Kriitiline koht on 9 ja 10 vahel.** Kes sinna jõuab, jõuab lõpuni.

**Märkmete repo.** Git või wiki, tähistuskokkulepe ja märkmikud sees. See on ainus asi, mis teeb kordamisest korduvkasutatava vara – järgmine doktorant alustab sealt, kus eelmine lõpetas.

**Kaheksa kohtumise versioon.** 1+2 kokku (tähistus jääb, dimensioonitrenn lüheneb), 3 terveks, 5 terveks, 6 terveks, 7+8 kokku, 9 terveks, 10 terveks, 11+12 kokku.

---

# Allikad

**Videod**

- 3Blue1Brown, "Essence of Linear Algebra" (osad 1–5, 9, 13, 14)
- 3Blue1Brown, "Backpropagation calculus"
- 3Blue1Brown, "But what is a GPT?"
- 3Blue1Brown, "Attention in transformers"
- Steve Brunton, "SVD Overview"
- Andrej Karpathy, "Neural Networks: Zero to Hero" (micrograd)
- Andrej Karpathy, "Let's build GPT"

**Tekstid**

- Vaswani jt (2017), "Attention is All You Need", ptk 3.1–3.5
- Boyd & Vandenberghe, *Introduction to Applied Linear Algebra* (tasuta)
- Deisenroth jt, *Mathematics for Machine Learning* (tasuta)
- Võhandu, Kuusik, Lind (2018), *Monotoonsed süsteemid ja nende rakendused*, TalTechi digikogu

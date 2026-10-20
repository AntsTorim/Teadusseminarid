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

# Plokk I: geomeetria (kohtumised 1–4)

[[1. Maatriks kui teisendus]]
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

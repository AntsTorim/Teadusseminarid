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
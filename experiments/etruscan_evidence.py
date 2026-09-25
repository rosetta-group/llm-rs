"""Build a cited development evidence layer without rewriting frozen experiments."""

import argparse
import json
from pathlib import Path

from etruscan.evidence import sha,validate
from experiments.etruscan_scaffolding import write

OUT=Path("experiments/etruscan-evidence")
ROOT=Path("artifacts/etruscan-sources/evidence-audit")


def ref(source,locator):
    return {"source":source,"locator":locator}


def span(tokens,label,status,refs,note="",basis="analyst_alignment"):
    return {"tokens":tokens,"label":label,"status":status,"evidence":refs,"basis":basis,"note":note}


def issue(kind,note,refs=()):
    return {"kind":kind,"note":note,"evidence":list(refs)}


def sources():
    rows=[
      ("schulze1993","Britta Schulze-Thulin","Zur Wortstellung im Etruskischen","1993",
       "Studi Etruschi 58, 177-195","schulze-thulin-word-order.pdf",
       "https://www.studietruschi.org/wp-content/uploads/2021/06/SE58_11.pdf",
       "Independent publication; shares ET/Rix scholarship. Not an independent archaeological observation."),
      ("desimone1970","Carlo de Simone","I morfemi etruschi -ce (-ke) e -χe","1970",
       "Studi Etruschi 38, 115-139","de-simone-morphemes.pdf",
       "https://www.studietruschi.org/wp-content/uploads/2021/07/SE38_08.pdf",
       "Direct grammatical argument and discussion of competing interpretations; historical, not current consensus by itself."),
      ("lopez2012_14","Roberto López Montero","Etrusco *huσ, huσ(i)ur: ¿un préstamo del griego ὗς / ὑύς?","2012-2014 (volume label)",
       "Faventia 34-36, 111-120","hus-husiur.pdf",
       "https://ddd.uab.cat/pub/faventia/faventia_a2012-14v34-36/faventia_a2012-14v34-36p111.pdf",
       "Uses inherited epigraphic editions and Wallace; not independent of the wider scholarly tradition. Loanword hypothesis is not adopted."),
      ("rigobianco2025","Luca Rigobianco","La famiglia etrusca nelle fonti epigrafiche di età classica","2025",
       "In La famiglia etrusca / The Etruscan Family, 67-88; CNR, DOI 10.19282/famigliaetrusca2025",
       "famiglia-etrusca-2025.pdf",
       "https://www.cnr.it/sites/default/files/public/media/attivita/editoria/LA%20FAMIGLIA%20ETRUSCA%2025%20NOVEMBRE%20DIGITALE.pdf",
       "Newer synthesis; cites ET/CIE and earlier analyses. Full CNR volume pinned; CC BY-SA 4.0 on imprint."),
      ("cie11","Corpus Inscriptionum Etruscarum","Volume I, inscription 11, Faesulae section",None,
       "Printed page 10, entry 11; section scan containing entries 1-474","cie-1-474.pdf",
       "https://www.studietruschi.org/wp-content/uploads/2025/01/CIE-I_tit.1_474.pdf",
       "Historical edition based on earlier copies, not a newly inspected object. Exact edition year not verified from this section scan."),
    ]
    out=[]
    for sid,author,title,year,publication,name,url,limits in rows:
        p=ROOT/name
        out.append({"id":sid,"author":author,"title":title,"year":year,"publication":publication,
                    "url":url,"local_path":str(p),"sha256":sha(p),"bytes":p.stat().st_size,
                    "retrieved":"2026-09-24","independence_limits":limits})
    out.append({"id":"met2013","author":"Theresa Huntsman","title":"Etruscan Language and Inscriptions",
                "year":"2013","publication":"The Metropolitan Museum of Art, Heilbrunn Timeline",
                "url":"https://www.metmuseum.org/essays/etruscan-language-and-inscriptions",
                "local_path":None,"sha256":None,"retrieved":"2026-09-24",
                "independence_limits":"Institutional account of object 26.60.94; further reading includes Wallace and Bonfante. Web access only, no local page snapshot."})
    return out


def build():
    originals={}
    for directory in ("etruscan-scaffolding","etruscan-graphs","etruscan-clauses"):
        path=Path("experiments",directory,"manifest.json")
        for record in json.loads(path.read_text()):
            originals[record["id"]]=(record,path)
    records=[]
    def add(tid,status,annotations,relations,issues):
        old,path=originals[tid]
        rs=[]
        for frame,tokens,confidence,refs in relations:
            rs.append({"frame":frame,"tokens":tokens,"status":confidence,"evidence":refs,
                       "basis":"analyst_alignment" if confidence=="corroborated" else "legacy_or_alternative_reading"})
        records.append({"id":tid,"status":status,"tokens":old["tokens"],"entities":old["entities"],
           "original":{"manifest_path":str(path),"manifest_sha256":sha(path),"gold":old["gold"]},
           "expert_review":"pending","fresh_evaluation_eligible":False,
           "text_scope":"the frozen dataset excerpt; not necessarily the complete monument",
           "spans":annotations,"relations":rs,"issues":issues})

    st182=[ref("schulze1993","p. 182, example 15; PDF page 6")]
    lm113=[ref("lopez2012_14","p. 113, Ta 1.168; PDF page 3")]
    add("Ta 1.168","corroborated",[
        span([4,5],"SPOUSE_PREDICATE","corroborated",st182),
        span([6],"DEATH_OUTSIDE_GRAPH","corroborated",lm113),
        span([7,8],"AGE_OUTSIDE_GRAPH","corroborated",lm113,"Numerical value is not endorsed."),
        span([9,10,11],"UNNAMED_CHILDREN_OUTSIDE_GRAPH","corroborated",lm113)],
        [(["SPOUSE_OF","e0","e1"],[4,5],"corroborated",st182)],
        [issue("reading_variant","Legacy N:xxii differs from the damaged -XII reading; preserve uncertainty, not an exact age.",lm113)])

    rig69=[ref("rigobianco2025","pp. 68-69, CIE 6213 = ET Cr 5.2; full-volume PDF pages 69-70")]
    st179=[ref("schulze1993","p. 179, SOV example Cr 5.2; PDF page 3")]
    add("Cr 5.2","corroborated",[
        span([5],"PLURAL_CHILD_PREDICATE","corroborated",rig69),
        span([9],"CONSTRUCTION_PREDICATE","corroborated",rig69+st179),
        span([7,8],"CONSTRUCTION_OBJECT","corroborated",rig69)],
        [(["CHILD_OF",e,"e2"],[5],"corroborated",rig69) for e in ("e0","e1")]
        +[(["MADE",e,"OBJECT"],[9],"corroborated",rig69+st179) for e in ("e0","e1")],
        [issue("excerpt_only","The published inscription continues beyond the legacy excerpt; graph completeness applies only to that excerpt.",rig69),
         issue("ontology_coarsening","MADE includes commissioning construction, not necessarily personal manual labour.",rig69)])

    de116=[ref("desimone1970","p. 116, TLE 868; PDF page 2")]
    add("Cr 3.20","corroborated",[
        span([4],"TRANSFER_PREDICATE","corroborated",de116),
        span([1],"DONOR_ROLE","corroborated",de116),
        span([2,3],"RECIPIENT_ROLE","corroborated",de116)],
        [(["TRANSFER","e0","OBJECT","e1"],[4],"corroborated",de116)],
        [issue("scope","This explicit active formula does not settle the agent/recipient interpretation of every mulu plus -si formula.",de116)])

    st190=[ref("schulze1993","p. 190, section 6.2.3.1, Ve 3.2 under pronoun-as-object SOV; PDF page 14")]
    add("Ve 3.2","partially_corroborated",[
        span([4],"OBJECT_PRONOUN","corroborated",st190),
        span([5],"TRANSFER_PREDICATE","corroborated",st190)],
        [(f,[5],"carried_forward",[]) for f in originals["Ve 3.2"][0]["gold"]],
        [issue("representation_collision","Keep mene distinct from the training men maker token; the three-letter hash conflated them.",st190),
         issue("review_needed","The two-donor graph is retained from Larth; this audit corroborates the construction, not a separate expert adjudication of every name role.")])

    de769=[ref("desimone1970","p. 116 TLE 769, pp. 119-120 note 14; PDF pages 2, 5-6")]
    met=[ref("met2013","Speaking-objects paragraph, alabastron 26.60.94")]
    add("Cr 3.18","disputed",[
        span([2],"GIFT_EXPRESSION","corroborated",de769+met),
        span([1,3],"AGENT_OR_RECIPIENT_ROLE","disputed",de769)],
        [(["TRANSFER","e0","OBJECT","UNSPECIFIED"],[2],"disputed",de769+met),
         (["TRANSFER","UNSPECIFIED","OBJECT","e0"],[2],"disputed",de769)],
        [issue("role_ambiguity","Alternative graphs are hypotheses, not two simultaneous gifts or newly adjudicated gold.",de769),
         issue("source_precision","Museum wording associates the gift with Licinius; it does not by itself formally resolve agent versus recipient.",met)])

    rig74=[ref("rigobianco2025","p. 74 note 44, CIE 11 = ET Vt 1.58; full-volume PDF page 75")]
    cie=[ref("cie11","printed p. 10, entry 11 and apparatus; PDF page 4")]
    add("Vt 1.58","disputed",[
        span([1,2,3,4],"NAME_FORMULA_UNRESOLVED","disputed",rig74+cie)],
        [(f,[],"disputed",rig74) for f in originals["Vt 1.58"][0]["gold"]],
        [issue("interpretation_ambiguity","The recent study explicitly flags the name formula as non-univocal.",rig74),
         issue("reading_variant","CIE preserves and discusses competing name readings; retain legacy tokens only as a versioned transcription.",cie)])

    av=[ref("schulze1993","p. 190 section 6.3.1.1.2 and note 6; PDF page 14")]
    add("AV 6.1","disputed",[
        span([2],"MAKING_PREDICATE","corroborated",av),
        span([3],"SURNAME_OR_SECOND_PARTICIPANT","disputed",av)],
        [(f,[2],"disputed",av) for f in originals["AV 6.1"][0]["gold"]],
        [issue("entity_anchor_ambiguity","The source allows titenas as another object unless it is the subject's family name; the old single-person anchor is not independently settled.",av)])

    for tid in ("Ru 5.1","Vs 1.28"):
        add(tid,"unverified",[],[(f,[],"carried_forward",[]) for f in originals[tid][0]["gold"]],
            [issue("source_gap","No independent passage settling the complete role graph was located in this bounded search. Legacy interpretation is neither confirmed nor rejected.")])
    bundle={"schema_version":1,"created":"2026-09-24","purpose":"development_evidence_only",
            "annotation_author":"Codex source audit; no external expert review",
            "sources":sources(),"records":records}
    summary=validate(bundle)
    write(OUT/"annotations.json",bundle)
    write(OUT/"validation.json",{"annotation_sha256":sha(OUT/"annotations.json"),"summary":summary})
    print(json.dumps(summary,indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("action",choices=("build","verify"))
    if p.parse_args().action=="build":
        build()
    else:
        bundle=json.loads((OUT/"annotations.json").read_text())
        saved=json.loads((OUT/"validation.json").read_text())
        if saved["annotation_sha256"]!=sha(OUT/"annotations.json"):
            raise ValueError("Annotation bytes changed")
        if saved["summary"]!=validate(bundle):
            raise ValueError("Validation summary changed")
        print(json.dumps(saved["summary"],indent=2))


if __name__=="__main__":
    main()

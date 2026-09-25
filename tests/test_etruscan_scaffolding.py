import copy
import random
import unittest

import pandas as pd

from etruscan import scaffolding as model
from experiments.etruscan_scaffolding import shuffle_frames


def view(tid, kinds):
    sequence, entities = [], {}
    for kind, case in kinds:
        if kind in ("W", "DEICTIC"):
            sequence.append({"kind":kind})
        else:
            eid=f"e{len(entities)}"
            attrs={"kind":kind,"case":case,"gender":"?"}
            entities[eid]=attrs
            sequence.append(dict(attrs,entity=eid))
    return {"id":tid,"sequence":sequence,"entities":entities}


class ScaffoldingTest(unittest.TestCase):
    def test_hidden_translation_and_word_spelling_do_not_reach_model(self):
        record={"id":"test", "tokens":["a","sech","b"],
                "entities":[{"tokens":[0],"kind":"PERSON"},{"tokens":[2],"kind":"PERSON"}],
                "translation":"daughter", "gold":[["CHILD_OF","e0","e1"]],
                "targets":[["CHILD_OF","e0","e1"]]}
        original=model.public_view(record,{})
        changed=copy.deepcopy(record)
        changed.update(translation="SECRET", gold=[], targets=[])
        self.assertEqual(original,model.public_view(changed,{}))
        self.assertEqual(set(original),{"id","sequence","entities","probes"})
        self.assertNotIn("sech",str(original))

    def test_dictionary_is_names_only_and_conflicts_stay_unknown(self):
        table=pd.DataFrame([
            {"Etruscan":"larthia","POS":"masc prae 2nd gen","Is suffix":False,"Is inferred":False},
            {"Etruscan":"larthia","POS":"fem prae nom acc","Is suffix":False,"Is inferred":False},
            {"Etruscan":"sech","POS":"nom acc","Is suffix":False,"Is inferred":False},
        ])
        morph=model.name_morphology(table)
        self.assertNotIn("sech",morph)
        public=model.public_view({"id":"a","tokens":["larthia"],
                                  "entities":[{"tokens":[0],"kind":"PERSON"}]},morph)
        self.assertEqual(public["entities"]["e0"]["gender"],"?")
        self.assertEqual(public["entities"]["e0"]["case"],"?")

    def test_directed_roles_are_not_interchangeable(self):
        frame=["TRANSFER","e0","OBJECT","e1"]
        self.assertEqual(model.transfer(frame,{"e0":"e1","e1":"e0"}),
                         ("TRANSFER","e1","OBJECT","e0"))
        self.assertIsNone(model.transfer(frame,{"e0":"e0"}))
        self.assertIsNone(model.transfer(["CHILD_OF","e0","e1"],{"e0":"e0","e1":"e0"}))
        self.assertEqual(model.frame_key(["SPOUSE_OF","e1","e0"]),
                         model.frame_key(["SPOUSE_OF","e0","e1"]))

    def test_synthetic_positive_control_recovers_roles_across_new_names(self):
        # Pure structural transfer; query has no gold or relation annotations.
        skeleton=[("PERSON","DIRECT"),("W",None),("DEITY","GEN")]
        train=[{"public":view(str(i),skeleton),"frames":[["TRANSFER","e0","OBJECT","e1"]]} for i in range(3)]
        prediction=model.predict(train,view("new",skeleton))
        self.assertEqual(prediction["accepted"],[["TRANSFER","e0","OBJECT","e1"]])

    def test_conflicting_templates_cause_abstention(self):
        skeleton=[("PERSON","DIRECT"),("PERSON","GEN"),("W",None)]
        train=[{"public":view("a",skeleton),"frames":[["CHILD_OF","e0","e1"]]},
               {"public":view("b",skeleton),"frames":[["SPOUSE_OF","e0","e1"]]}]
        self.assertEqual(model.predict(train,view("new",skeleton))["accepted"],[])

    def test_discontinuous_name_remains_one_entity(self):
        record={"id":"a","tokens":["first","verb","last"],
                "entities":[{"tokens":[0,2],"kind":"PERSON"}]}
        public=model.public_view(record,{})
        self.assertEqual(len(public["entities"]),1)
        self.assertEqual([t.get("entity") for t in public["sequence"]],["e0",None,"e0"])
        distance,mapping=model.align(public,dict(public,id="b"))
        self.assertEqual((distance,mapping),(0.0,{"e0":"e0"}))

    def test_train_test_monument_collision_is_rejected(self):
        public=view("same",[("PERSON","DIRECT")])
        with self.assertRaises(ValueError):
            model.predict([{"public":public,"frames":[]}],public)

    def test_translation_shuffle_preserves_participant_count_and_graph_inventory(self):
        train=[{"public":view("a",[("PERSON","DIRECT")]),"frames":[["MADE","e0","OBJECT"]]},
               {"public":view("b",[("PERSON","GEN")]),"frames":[["OWNED_BY","OBJECT","e0"]]},
               {"public":view("c",[("PERSON","DIRECT"),("PERSON","GEN")]),"frames":[["CHILD_OF","e0","e1"]]}]
        shuffled=shuffle_frames(train,random.Random(1))
        self.assertEqual(sorted(str(r["frames"]) for r in shuffled),sorted(str(r["frames"]) for r in train))
        self.assertEqual(shuffled[2]["frames"],train[2]["frames"])
        for before,after in zip(train,shuffled):
            self.assertEqual(before["public"],after["public"])

    def test_joint_positive_control_and_probe_opacity(self):
        skeleton=[("PERSON","DIRECT"),("W",None),("DEITY","GEN")]
        train=[{"public":view(str(i),skeleton),"frames":[["TRANSFER","e0","OBJECT","e1"]]} for i in range(3)]
        queries=[dict(view("new"+str(i),skeleton),probes=["opaque-root"]) for i in range(2)]
        predictions,families=model.predict_joint(train,queries)
        for r in predictions.values():
            self.assertEqual(r["accepted"],[["TRANSFER","e0","OBJECT","e1"]])
        self.assertEqual(families["opaque-root"]["winner"],"TRANSFER")

    def test_joint_conflicting_relation_evidence_abstains(self):
        skeleton=[("PERSON","DIRECT"),("PERSON","GEN"),("W",None)]
        train=[{"public":view("a",skeleton),"frames":[["CHILD_OF","e0","e1"]]},
               {"public":view("b",skeleton),"frames":[["SPOUSE_OF","e0","e1"]]}]
        predictions,_=model.predict_joint(train,[dict(view("new",skeleton),probes=["opaque"])])
        self.assertEqual(predictions["new"]["accepted"],[])


if __name__=="__main__":
    unittest.main()

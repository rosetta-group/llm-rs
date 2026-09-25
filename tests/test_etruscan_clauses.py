import copy
import unittest

from etruscan import clause_coverage as model, scaffolding as old


def view(tid,n,stems=()):
    entities={f"e{i}":{"kind":"PERSON","case":"?","gender":"?"} for i in range(n)}
    return {"id":tid,"entities":entities,
            "sequence":[dict(a,entity=e) for e,a in entities.items()]+[{"kind":"W","lexeme":s} for s in stems]}


def training():
    return ([{"public":view("made",1,["make"]),"frames":[["MADE","e0","OBJECT"]]}]
        +[{"public":view(f"child{i}",2,["parent"]),"frames":[["CHILD_OF","e0","e1"]]} for i in range(3)]
        +[{"public":view(f"own{i}",1,["vase"]),"frames":[["OWNED_BY","OBJECT","e0"]]} for i in range(3)])


class ClauseCoverageTest(unittest.TestCase):
    def test_predicate_alignment_uses_training_only(self):
        learned=model.fit(training())
        self.assertEqual(learned["lexicon"]["make"]["relation"],"MADE")
        self.assertGreaterEqual(learned["lexicon"]["make"]["support"],model.MIN_ANCHOR_SUPPORT)
        self.assertEqual(model.anchors(view("q",1,["unseen"]),learned),[])

    def test_duplicate_stem_counts_once_per_monument(self):
        before=training()
        after=copy.deepcopy(before)
        after[0]["public"]["sequence"].append({"kind":"W","lexeme":"make"})
        self.assertEqual(model.fit(before)["lexicon"],model.fit(after)["lexicon"])

    def test_coverage_recovers_omitted_construction(self):
        query=view("q",2,["make"])
        learned=model.fit(training())
        learned["bag"]={}  # Isolate span coverage from the older bag-of-stems reward.
        edges={("CHILD_OF","e0","e1"):{"cost":0.0},("MADE","e0","OBJECT"):{"cost":0.0}}
        complete=model.search(edges,query,learned)
        incomplete=model.search(edges,query,learned,coverage=False)
        self.assertEqual(set(map(tuple,complete["accepted"])),set(edges))
        self.assertEqual(incomplete["best_graph"],[["CHILD_OF","e0","e1"]])

    def test_repeated_predicate_cannot_reuse_one_edge(self):
        self.assertEqual(model.matching_cost(((0.0,),(0.0,))),1.0)
        self.assertEqual(model.matching_cost(((0.0,1.0),(1.0,0.0))),0.0)

    def test_local_roles_distinguish_reversed_parentage(self):
        q=view("q",2,["parent"])
        learned=model.fit(training())
        a={"position":2,"stem":"parent","relation":"CHILD_OF"}
        correct=model.attachment_cost(q,("CHILD_OF","e0","e1"),a,learned)
        wrong=model.attachment_cost(q,("CHILD_OF","e1","e0"),a,learned)
        self.assertLess(correct,wrong)
        self.assertEqual(model.attachment_cost(q,("CHILD_OF","e1","e0"),a,learned,local=False),0)

    def test_mismatched_relation_does_not_explain_span(self):
        self.assertEqual(model.attachment_cost(view("q",1),["MADE","e0","OBJECT"],
                         {"relation":"OWNED_BY"},{},local=False),1.0)

    def test_four_edges_can_be_composed(self):
        q=view("q",4)
        edges={("MADE",f"e{i}","OBJECT"):{"cost":0.0} for i in range(4)}
        result=model.search(edges,q,{"lexicon":{},"templates":{},"bag":{}})
        self.assertEqual(set(map(tuple,result["accepted"])),set(edges))

    def test_test_translation_and_gold_do_not_reach_model(self):
        record={"id":"q","tokens":["name","word"],"entities":[{"tokens":[0],"kind":"PERSON"}],
                "translation":"made","gold":[["MADE","e0","OBJECT"]]}
        before=old.public_view(record,{})
        record.update(translation="SECRET",gold=[])
        after=old.public_view(record,{})
        self.assertEqual(before,after)
        self.assertEqual(model.predict(training(),[before]),model.predict(training(),[after]))

    def test_same_monument_is_rejected(self):
        with self.assertRaises(ValueError):
            model.predict(training(),[training()[0]["public"]])

    def test_training_labels_change_lexicon_and_attachment_templates(self):
        train=training()
        changed=copy.deepcopy(train)
        changed[0]["frames"]=[["TRANSFER","e0","OBJECT","UNSPECIFIED"]]
        self.assertNotEqual(model.fit(train),model.fit(changed))
        self.assertEqual(model.fit(changed)["lexicon"]["make"]["relation"],"TRANSFER")


if __name__=="__main__":
    unittest.main()

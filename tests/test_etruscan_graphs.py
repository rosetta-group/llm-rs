import copy
import unittest

from etruscan import graph_alignment as model, scaffolding as old


def view(tid, cases, stem=None):
    entities={f"e{i}":{"kind":"PERSON","case":case,"gender":"?"} for i,case in enumerate(cases)}
    sequence=[dict(attrs,entity=e) for e,attrs in entities.items()]
    if stem:
        sequence.append({"kind":"W","lexeme":stem})
    return {"id":tid,"entities":entities,"sequence":sequence}


class GraphAlignmentTest(unittest.TestCase):
    def test_all_injective_assignments_including_reversed_order(self):
        options=model.mappings(view("a",["DIRECT","GEN"]),view("b",["GEN","DIRECT","GEN"]))
        self.assertEqual(len(options),6)
        self.assertIn({"e0":"e1","e1":"e0"},[o["mapping"] for o in options])
        self.assertTrue(all(len(set(o["mapping"].values()))==2 for o in options))

    def test_scope_and_overlap_are_enforced(self):
        with self.assertRaises(ValueError):
            model.mappings(view("a",["?"]),view("b",["?"]*5))
        public=view("a",["?"])
        with self.assertRaises(ValueError):
            model.geometry([{"public":public,"frames":[]}],[public])

    def test_complete_search_cannot_ignore_second_parent(self):
        query=view("q",["DIRECT","GEN","GEN"])
        edges={("CHILD_OF","e0","e1"):{"cost":0.0},
               ("CHILD_OF","e0","e2"):{"cost":0.04}}
        complete=model.search(edges,query,{})
        partial=model.search(edges,query,{},cover_all=False)
        self.assertEqual(set(map(tuple,complete["accepted"])),set(edges))
        self.assertEqual(len(partial["best_graph"]),1)
        self.assertTrue(model.connected(complete["accepted"],query["entities"]))

    def test_two_recipients_connect_through_object(self):
        graph=(("TRANSFER","UNSPECIFIED","OBJECT","e0"),
               ("TRANSFER","UNSPECIFIED","OBJECT","e1"))
        self.assertTrue(model.valid(graph))
        self.assertTrue(model.connected(graph,{"e0","e1"}))
        self.assertFalse(model.connected(graph,{"e0","e1","e2"}))

    def test_constraints_reject_cycles_conflicts_and_disconnection(self):
        self.assertFalse(model.valid((("CHILD_OF","e0","e1"),("CHILD_OF","e1","e0"))))
        self.assertFalse(model.valid((("CHILD_OF","e0","e1"),("SPOUSE_OF","e0","e1"))))
        self.assertFalse(model.valid((("MADE","e0","OBJECT"),("OWNED_BY","OBJECT","e1"))))
        self.assertFalse(model.connected((("CHILD_OF","e0","e1"),("CHILD_OF","e2","e3")),{"e0","e1","e2","e3"}))
        self.assertTrue(model.valid((("CHILD_OF","e0","e1"),("MADE","e0","OBJECT"))))

    def test_tied_complete_graphs_abstain(self):
        edges={("MADE","e0","OBJECT"):{"cost":0.0},("OWNED_BY","OBJECT","e0"):{"cost":0.0}}
        result=model.search(edges,view("q",["?"]),{})
        self.assertEqual(result["accepted"],[])
        self.assertEqual(result["margin"],0.0)

    def test_positive_control_reverses_roles_using_name_case(self):
        train=[{"public":view("train",["DIRECT","GEN"]),"frames":[["CHILD_OF","e0","e1"]]}]
        result=model.predict(train,[view("query",["GEN","DIRECT"])])["query"]
        self.assertEqual(result["accepted"],[["CHILD_OF","e1","e0"]])

    def test_geometry_stays_fixed_but_shuffled_labels_change_evidence(self):
        train=[{"public":view("train",["DIRECT"],"opaque"),"frames":[["MADE","e0","OBJECT"]]}]
        changed=copy.deepcopy(train)
        changed[0]["frames"]=[["OWNED_BY","OBJECT","e0"]]
        queries=[view("query",["DIRECT"],"opaque")]
        cache=model.geometry(train,queries)
        self.assertEqual(cache,model.geometry(changed,queries))
        self.assertNotEqual(model.lexical_evidence(train),model.lexical_evidence(changed))
        self.assertEqual(model.predict(changed,queries,cached=cache),model.predict(changed,queries))

    def test_hidden_translation_and_answers_cannot_change_prediction(self):
        record={"id":"q","tokens":["a","mulu"],"entities":[{"tokens":[0],"kind":"PERSON"}],
                "gold":[["MADE","e0","OBJECT"]],"translation":"made"}
        before=old.public_view(record,{})
        record.update(gold=[["OWNED_BY","OBJECT","e0"]],translation="SECRET")
        after=old.public_view(record,{})
        self.assertEqual(before,after)
        self.assertNotIn("mulu",str(after))
        train=[{"public":view("train",["?"]),"frames":[["MADE","e0","OBJECT"]]}]
        self.assertEqual(model.predict(train,[before]),model.predict(train,[after]))


if __name__=="__main__":
    unittest.main()

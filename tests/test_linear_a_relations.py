import copy
import unittest
from linear_a import relation_control as c


def row(key, obj, sequence=('N','W001')):
    return {'id':key, 'object':obj, 'sequence':list(sequence)}


def sample(word='person', relation='i-*65', status='scored'):
    raw=f'{word} {relation}'
    parts=[]
    for term,kind in [(word,'name'),(relation,'word')]:
        start=raw.index(term)
        parts.append({'text':term,'kind':kind,'segments':[{'line_index':1,'raw':raw,
                     'span':[start,start+len(term)],'text':term}]})
    return [{'id':'x','object':'1','status':status,'parts':parts}], {'1':{'content':raw}}


class RelationControlTests(unittest.TestCase):
    def test_damaged_candidates_are_not_intact_hits_or_substrings(self):
        result=c.inventory({'1':{'heading_short':'test','content':'i-*65 i-*65-qe i-*6̣5̣ di-*65-pa-ta ra-]ke-da-mo-ni-jo-u-jo'}})
        self.assertEqual([r['token'] for r in result['hits'] if r['status']=='intact'],['i-*65','i-*65-qe'])
        self.assertEqual(sum(r['status']=='uncertain_text' for r in result['hits']),2)

    def test_multiline_erasure_not_retrieved(self):
        result=c.inventory({'1':{'heading_short':'test','content':'⟦i-*65\ni-jo⟧\ntu-ka-te-qe'}})
        self.assertEqual([r['token'] for r in result['hits']],['tu-ka-te-qe'])
        self.assertEqual(result['hits'][0]['line_index'],3)

    def test_source_join_keeps_patronymic_onomastic(self):
        records={'1':{'content':'child parent\ni-jo'}}
        def segment(n,raw,start,text):return {'line_index':n,'raw':raw,'span':[start,start+len(text)],'text':text}
        cases=[{'id':'x','object':'1','status':'scored','parts':[
            {'text':'child','kind':'name','segments':[segment(1,'child parent',0,'child')]},
            {'text':'parent-i-jo','kind':'name','segments':[segment(1,'child parent',6,'parent'),segment(2,'i-jo',0,'i-jo')]}]}]
        public,vocab=c.public_view(cases,records)
        self.assertEqual(public[0]['sequence'],['N','N']);self.assertEqual(vocab,{})
        broken=copy.deepcopy(cases);broken[0]['parts'].reverse()
        with self.assertRaises(ValueError):c.public_view(broken,records)

    def test_unresolved_and_uncertain_markers_never_enter_public_view(self):
        cases,records=sample(status='unresolved')
        self.assertEqual(c.public_view(cases,records),([],{}))
        cases,records=sample(relation='i-*6̣5̣')
        with self.assertRaises(ValueError):c.public_view(cases,records)

    def test_names_and_meaningless_word_codes_do_not_change_predictions(self):
        a,ra=sample('ko-pe-re-u');b,rb=sample('u-re-pe-ko')
        self.assertEqual(c.public_view(a,ra)[0],c.public_view(b,rb)[0])
        train=[row('a','a'),row('b','b')];labels={'a':'FAMILY','b':'FAMILY'}
        p=c.predict(train,labels,row('test','heldout'),'frame_marker')
        rename=[row('a','a',('N','W987')),row('b','b',('N','W987'))]
        self.assertEqual(p,c.predict(rename,labels,row('test','heldout',('N','W987')),'frame_marker'))

    def test_whole_object_holdout_removes_both_class_examples(self):
        rows=[row('family','one'),row('pair','one'),row('else','two',('W999','N'))]
        labels={'family':'FAMILY','pair':'OTHER','else':'OTHER'}
        predictions=c.evaluate(rows,labels,'frame_marker')['predictions']
        for key in ('family','pair'):
            self.assertEqual(predictions[key]['support_objects'],[])
            self.assertIsNone(predictions[key]['prediction'])

    def test_long_roster_is_one_vote(self):
        rows=[row(str(i),'one') for i in range(50)]+[row('other','two')]
        labels={str(i):'FAMILY' for i in range(50)};labels['other']='OTHER'
        pred=c.predict(rows,labels,row('test','heldout'),'frame_marker')
        self.assertEqual(pred['votes'],{'FAMILY':1,'OTHER':1})
        self.assertIsNone(pred['prediction'])

    def test_family_recall_cannot_be_replaced_by_negative_accuracy(self):
        rows=[row('f','f'),row('o','o')];labels={'f':'FAMILY','o':'OTHER'}
        preds={'f':{'prediction':None},'o':{'prediction':'OTHER'}}
        m=c.metrics(rows,labels,preds)
        self.assertEqual(m['conditional_accuracy'],1)
        self.assertEqual(m['family_recall'],0)
        self.assertFalse(c.performance_pass(m))
        preds['o']['prediction']='FAMILY'
        self.assertEqual(c.metrics(rows,labels,preds)['false_family_rate'],1)

    def test_marker_identity_separates_a_frame_collision_but_not_names(self):
        rows=[row('f','f'),row('o','o',('N','W002'))]
        labels={'f':'FAMILY','o':'OTHER'}
        self.assertEqual(c.ceiling(rows,labels,'frame')['balanced_recall_upper_bound'],.5)
        self.assertEqual(c.ceiling(rows,labels,'frame_marker')['balanced_recall_upper_bound'],1)


if __name__=='__main__':unittest.main()

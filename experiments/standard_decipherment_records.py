"""Commit the exact challenge and predictions only after frozen grading."""
import argparse
import gzip
import tarfile

from experiments.standard_decipherment import ROOT, STATE, OUT, read, write, verify
from voynich.data import digest

FILES = ['public.json','challenge.json','predictions.json','segmentation-predictions.json','evaluator-only/answers.json']


def pack():
    verify()
    results=read(OUT/'results.json')
    if results['predictions_sha256']!=digest(STATE/'predictions.json'):
        raise ValueError('Only graded, unchanged predictions may be archived')
    if read(OUT/'verification.json')['status']!='passed':
        raise ValueError('Run the independent audit before release')
    target=OUT/'evaluated-records.tar.gz'
    with target.open('wb') as raw, gzip.GzipFile(fileobj=raw,mode='wb',mtime=0,filename='') as zipped:
        with tarfile.open(fileobj=zipped,mode='w') as archive:
            for name in FILES:
                path=STATE/name
                info=archive.gettarinfo(str(path),arcname=name)
                info.uid=info.gid=info.mtime=0;info.uname=info.gname=''
                with path.open('rb') as stream:
                    archive.addfile(info,stream)
    write(OUT/'records.json',dict(archive_sha256=digest(target),files={p:digest(STATE/p) for p in FILES},
        release='After grading only. Plaintext references are now disclosed; never reuse these passages for tuning or a fresh test.',
        license='ISDT-derived passages: CC BY-NC-SA 3.0; Italian-Old/Dante annotations: CC BY-SA 4.0. See SOURCE_LICENSE.md.'))
    print('Archived the graded challenge, references, seeds and predictions')


def restore():
    manifest=read(OUT/'records.json');archive=OUT/'evaluated-records.tar.gz'
    if digest(archive)!=manifest['archive_sha256']: raise ValueError('Archive drift')
    STATE.mkdir(parents=True,exist_ok=True)
    with tarfile.open(archive,'r:gz') as bundle:
        for member in bundle.getmembers():
            target=STATE/member.name
            if target.exists():
                if digest(target)!=manifest['files'][member.name]: raise ValueError('Existing record differs')
                continue
            bundle.extract(member,STATE,filter='data')
    for p,h in manifest['files'].items():
        if digest(STATE/p)!=h: raise ValueError('Restored record differs')
    print('Exact evaluated records restored; these cases are disclosed')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['pack','restore'])
    globals()[parser.parse_args().command]()

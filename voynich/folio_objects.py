"""Validate reviewed object observations; derive transparent multi-label descriptions."""
KINDS = {'whole_plant', 'plant_part', 'leaf', 'root', 'flower_like', 'human_figure', 'animal',
         'vessel', 'basin', 'channel', 'circular_structure', 'radial_motif', 'face',
         'star_like_mark', 'decorative_mark', 'uncertain_form'}
PREDICATES = {'inside', 'part_of', 'connected_to', 'above', 'left_of', 'overlaps'}
BOTANICAL = {'whole_plant', 'plant_part', 'leaf', 'root', 'flower_like'}
DOMAINS = ('botanical', 'human_figures', 'animal_figures', 'containers', 'basins_channels',
           'diagrammatic', 'celestial_like', 'text_dominant', 'sparse_marks')
PATTERNS = ('linked_basins', 'figures_in_linked_basins', 'vessels_beside_botanical_groups',
            'animal_in_circular_diagram', 'linked_circular_structures')


def validate(row):
    objects = row['objects']; ids = {o['id'] for o in objects}
    if len(ids) != len(objects):
        raise ValueError('Duplicate object ID')
    for o in objects:
        if o['kind'] not in KINDS or o['certainty'] not in ('clear', 'tentative'):
            raise ValueError('Unknown object kind/certainty')
        x, y, r, b = o['bbox']
        if not 0 <= x < r <= 1 or not 0 <= y < b <= 1:
            raise ValueError('Invalid evidence rectangle')
        if not isinstance(o['count_min'], int) or o['count_min'] < 1 or (o['count_max'] is not None and
            (not isinstance(o['count_max'], int) or o['count_max'] < o['count_min'])):
            raise ValueError('Invalid count bound')
    for r in row['relations']:
        if r['subject'] not in ids or r['object'] not in ids or r['predicate'] not in PREDICATES:
            raise ValueError('Invalid relation')
        if r['certainty'] not in ('clear', 'tentative'):
            raise ValueError('Invalid relation certainty')
    if row['layout'] not in ('illustrated', 'text_dominant', 'sparse'):
        raise ValueError('Invalid layout')


def features(row, clear_only=False):
    objects = {o['id']: o for o in row['objects'] if not clear_only or o['certainty'] == 'clear'}
    kinds = {o['kind'] for o in objects.values()}
    relations = [r for r in row['relations'] if r['subject'] in objects and r['object'] in objects
                 and (not clear_only or r['certainty'] == 'clear')]
    domains = set(); patterns = set()
    if kinds & BOTANICAL: domains.add('botanical')
    for kind, domain in [('human_figure', 'human_figures'), ('animal', 'animal_figures'), ('vessel', 'containers')]:
        if kind in kinds: domains.add(domain)
    if kinds & {'basin', 'channel'}: domains.add('basins_channels')
    if kinds & {'circular_structure', 'radial_motif'}: domains.add('diagrammatic')
    if 'diagrammatic' in domains and kinds & {'star_like_mark', 'face'}: domains.add('celestial_like')
    if row['layout'] == 'text_dominant': domains.add('text_dominant')
    if row['layout'] == 'sparse': domains.add('sparse_marks')
    links = {}; occupied = set()
    for r in relations:
        a, b, p = r['subject'], r['object'], r['predicate']
        ka, kb = objects[a]['kind'], objects[b]['kind']
        if p == 'connected_to':
            if ka == 'channel' and kb == 'basin': links.setdefault(a, set()).add(b)
            if kb == 'channel' and ka == 'basin': links.setdefault(b, set()).add(a)
            if ka == kb == 'circular_structure' and (a != b or objects[a]['count_min'] >= 2):
                patterns.add('linked_circular_structures')
        if p == 'inside' and ka == 'human_figure' and kb == 'basin': occupied.add(b)
        if p == 'inside' and ka == 'animal' and kb == 'circular_structure': patterns.add('animal_in_circular_diagram')
        if p == 'left_of' and ka == 'vessel' and kb in BOTANICAL: patterns.add('vessels_beside_botanical_groups')
    for basins in links.values():
        if len(basins) >= 2: patterns.add('linked_basins')
        if len(basins & occupied) >= 2: patterns.add('figures_in_linked_basins')
    return sorted(domains), sorted(patterns)


def describe(row):
    validate(row)
    domains, patterns = features(row)
    clear_domains, clear_patterns = features(row, clear_only=True)
    names = ', '.join(sorted({o['kind'].replace('_', ' ') for o in row['objects']})) or 'no objects assigned'
    relation_text = '; '.join(f"{r['subject']} {r['predicate'].replace('_', ' ')} {r['object']}" for r in row['relations'])
    return dict(domains=domains, tentative_domains=sorted(set(domains) - set(clear_domains)),
                relation_patterns=patterns, tentative_patterns=sorted(set(patterns) - set(clear_patterns)),
                object_group_count=len(row['objects']), relation_count=len(row['relations']),
                description=names.capitalize() + ('. ' + relation_text if relation_text else '') + '. ' + row['notes'])

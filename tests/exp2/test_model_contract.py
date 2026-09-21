import pytest
import json
from appl.model_contract import edited_xml,parse_xml


def test_xml_batch_rejects_bad_xpath_without_partial_file_output():
    original='<mujoco><worldbody><geom name="a" size="1"/></worldbody></mujoco>'
    with pytest.raises(SyntaxError):
        edited_xml(original,[dict(operation='set',selector='.//geom',attributes={'size':'2'}),
                             dict(operation='set',selector='//geom',attributes={'size':'3'})])
    assert parse_xml(original).find('.//geom').get('size')=='1'
    changed=edited_xml(original,[dict(operation='set',selector='.//geom',attributes={'size':'2'})])
    assert parse_xml(changed).find('.//geom').get('size')=='2'


def test_external_model_execution_is_rejected():
    for text in ('<mujoco><include file="secret.xml"/></mujoco>',
                 '<mujoco><extension><plugin plugin="code"/></extension></mujoco>'):
        with pytest.raises(ValueError):parse_xml(text)


def test_calibration_contract_survives_json_roundtrip():
    from appl.journal import sha
    contract=dict(cases=[('demo1000',0),('demo1007',0)],limits=dict(red=.02))
    assert sha(contract)==sha(json.loads(json.dumps(contract)))
    assert sha(contract)!=sha(dict(contract,limits=dict(red=.2)))

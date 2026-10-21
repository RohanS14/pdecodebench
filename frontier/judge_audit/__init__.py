"""Human audit of the LLM judge used in §4.2 (frontier/judge_invalidity_reasoning.py).

  select_sample.py  draw 5 judged rows per thinking setting (seeded), write the private key
  build_packet.py   render what the expert sees (HTML packet, grading CSV, xlsx)
  score_audit.py    compare the expert's grades with the judge's
"""

# ------- lookup_ext (FUNCTION_CALL ONLY — no DIExpression fallback) -------
if tag == "function_call" and lower(a.get("name","")) == "lookup_ext":
    proj, job, df = context_for(e)
    schema_out = schema_out_from_DISchema(e, pm, cur_schema)

    # read ONLY attributes on the Function Call (authoritative)
    # e.g. <FUNCTION_CALL name="lookup_ext" type="DI"
    #       tableDatastore="DS_ODS" tableOwner="BRIOVIEW" tableName="BANK_CMMT_TYPE_GL_XREF" .../>
    dsx  = (a.get("tabledatastore") or a.get("tableDatastore") or "").strip()
    schx = (a.get("tableowner")     or a.get("tableOwner")     or "").strip()
    tbx  = (a.get("tablename")      or a.get("tableName")      or "").strip()

    if dsx and tbx and schema_out:
        k = (proj, job, df, _norm_key(dsx), _norm_key(schx), _norm_key(tbx))
        remember_display(dsx, schx, tbx)
        lookup_ext_pos[k].add(schema_out)
        seen_ext_keys.add(k)  # so nothing else can add a dup for this key



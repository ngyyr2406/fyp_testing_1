from __future__ import annotations

import io
import json
import os
import re
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from model_loader import load_all_models
from candidate_ranking import rank_candidates
from resume_matching import match_resume_to_job

ROOT = Path(__file__).resolve().parent
MODEL_ROOT = Path(os.environ.get("MODEL_ROOT", str(ROOT)))
EVAL_DIR = ROOT / "evaluation_results"
HP_DIR = ROOT / "hyperparameter_results"
DATA_DIR = ROOT / "app_data"
UPLOAD_DIR = DATA_DIR / "uploads"
DB_PATH = DATA_DIR / "history.sqlite3"
for p in (DATA_DIR, UPLOAD_DIR):
    p.mkdir(parents=True, exist_ok=True)

MODELS = ["TF-IDF", "BERT", "E5", "BGE"]
MODEL_DISPLAY = {"TF-IDF":"TF-IDF", "BERT":"BERT", "E5":"Fine-tuned E5", "BGE":"Fine-tuned BGE"}
METRICS = ["Accuracy", "Precision", "Recall", "F1-score", "MAP", "nDCG@10"]


# ------------------------- Persistent history --------------------------------
def connect_db():
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    return c

def init_db():
    with connect_db() as c:
        c.execute("""CREATE TABLE IF NOT EXISTS analyses(
            id TEXT PRIMARY KEY, created_at TEXT, kind TEXT, model TEXT,
            title TEXT, jd_text TEXT, result_json TEXT, files_json TEXT,
            shortlist_json TEXT)""")
        c.commit()

def save_record(kind, model, title, jd_text, result, files):
    rid = str(uuid.uuid4())
    with connect_db() as c:
        c.execute("INSERT INTO analyses VALUES(?,?,?,?,?,?,?,?,?)", (
            rid, datetime.now().isoformat(timespec="seconds"), kind, model,
            title, jd_text, json.dumps(result, ensure_ascii=False),
            json.dumps(files, ensure_ascii=False), "[]"))
        c.commit()
    return rid

def load_records(q=""):
    with connect_db() as c:
        if q.strip():
            pat=f"%{q.strip()}%"
            rows=c.execute("SELECT * FROM analyses WHERE title LIKE ? OR jd_text LIKE ? OR kind LIKE ? ORDER BY created_at DESC",(pat,pat,pat)).fetchall()
        else:
            rows=c.execute("SELECT * FROM analyses ORDER BY created_at DESC").fetchall()
    return [dict(r) for r in rows]

def delete_record(rid):
    with connect_db() as c:
        row=c.execute("SELECT files_json FROM analyses WHERE id=?",(rid,)).fetchone()
        if row:
            for f in json.loads(row["files_json"] or "[]"):
                p=Path(f.get("path",""))
                if p.exists() and UPLOAD_DIR in p.parents:
                    p.unlink()
        c.execute("DELETE FROM analyses WHERE id=?",(rid,))
        c.commit()

def set_shortlist(rid, names):
    with connect_db() as c:
        c.execute("UPDATE analyses SET shortlist_json=? WHERE id=?",(json.dumps(names),rid))
        c.commit()


# ------------------------------ File input ----------------------------------
def extract_text(upload):
    suffix=Path(upload.name).suffix.lower()
    raw=upload.getvalue()
    if suffix==".txt":
        return raw.decode("utf-8",errors="replace")
    if suffix==".pdf":
        from pypdf import PdfReader
        text="\n".join(p.extract_text() or "" for p in PdfReader(io.BytesIO(raw)).pages)
        if not text.strip():
            raise ValueError("This PDF has no selectable text. It may be scanned; OCR is not a built-in Streamlit feature.")
        return text
    if suffix==".docx":
        from docx import Document
        doc=Document(io.BytesIO(raw))
        parts=[p.text for p in doc.paragraphs]
        for table in doc.tables:
            for row in table.rows:
                parts.append(" | ".join(cell.text for cell in row.cells))
        return "\n".join(parts)
    if suffix==".doc":
        raise ValueError("Legacy .doc is optional; convert it to DOCX or PDF.")
    if suffix in {".jpg",".jpeg",".png"}:
        raise ValueError("Streamlit file_uploader does not perform OCR. JPG/PNG require an OCR service or library.")
    raise ValueError(f"Unsupported file format: {suffix}")

def save_upload(upload, rid):
    safe=re.sub(r"[^A-Za-z0-9._-]+","_",upload.name)
    dest=UPLOAD_DIR/f"{rid}_{safe}"
    dest.write_bytes(upload.getvalue())
    return {"name":upload.name,"path":str(dest),"bytes":len(upload.getvalue())}

def read_jd(upload, typed):
    return extract_text(upload).strip() if upload else (typed or "").strip()


# ------------------------------ Model loading -------------------------------
@st.cache_resource(show_spinner="Loading the selected final checkpoints...")
def get_models(model_root_string):
    return load_all_models(base_dir=Path(model_root_string))

def choose_model(key, default="E5"):
    return st.selectbox("Matching model", MODELS,
        index=MODELS.index(default), format_func=lambda x:MODEL_DISPLAY[x], key=key)


# ------------------------------ Evaluation ----------------------------------
def load_csv(path):
    return pd.read_csv(path) if path.exists() else None

def selected_config():
    p=EVAL_DIR/"final_selected_configurations.csv"
    df=load_csv(p)
    if df is not None:
        return df
    cfg=ROOT/"model_config.json"
    if cfg.exists():
        data=json.loads(cfg.read_text(encoding="utf-8"))
        return pd.DataFrame(data.get("selected_configurations",[]))
    return pd.DataFrame()

def hyperparameter_tables(model):
    folder=HP_DIR/model
    files=sorted(folder.glob("*hyperparameter*.csv"))
    frames=[]
    for f in files:
        d=pd.read_csv(f)
        d["Source file"]=f.name
        d["Experiment"]=f.stem.replace(model.lower().replace("-",""),"").strip("_") or f.stem
        frames.append(d)
    return pd.concat(frames,ignore_index=True) if frames else pd.DataFrame()

def evaluation_page():
    st.title("Model Evaluation")
    st.caption("Experiment comparison and selected-checkpoint performance.")

    summary=load_csv(EVAL_DIR/"final_model_summary.csv")
    if summary is None:
        st.warning("Add evaluation_results/final_model_summary.csv to display the final four-model comparison.")
    else:
        st.subheader("Final selected-model comparison")
        summary["Model"]=summary["Model"].replace({"E5":"Fine-tuned E5","BGE":"Fine-tuned BGE"})
        show_cols=["Model"]+[m for m in METRICS if m in summary.columns]
        st.dataframe(summary[show_cols].style.format({m:"{:.4f}" for m in METRICS if m in summary.columns}),use_container_width=True,hide_index=True)
        chart=summary.set_index("Model")[[m for m in METRICS if m in summary.columns]]
        st.bar_chart(chart,use_container_width=True)

    st.subheader("E5 hyperparameter experiments")
    e5=hyperparameter_tables("E5")
    if not e5.empty:
        # use authoritative comparison file if present to show explicit experiment labels and selected status
        cmp=load_csv(EVAL_DIR/"e5_experiment_comparison.csv")
        if cmp is not None:
            e5view=cmp.copy()
            e5view["Selection"]=e5view["Selection"].replace({"-":"Not selected"})
            st.dataframe(e5view,use_container_width=True,hide_index=True)
            st.bar_chart(cmp.groupby("Experiment")["Mean Validation F1"].first(),use_container_width=True)
        else:
            st.dataframe(e5,use_container_width=True,hide_index=True)
    else:
        st.info("No E5 hyperparameter CSV files found.")

    st.subheader("BGE hyperparameter experiments")
    bge=hyperparameter_tables("BGE")
    if not bge.empty:
        cmp=load_csv(EVAL_DIR/"bge_experiment_comparison.csv")
        if cmp is not None:
            eview=cmp.copy()
            eview["Selection"]=eview["Selection"].replace({"-":"Not selected"})
            st.dataframe(eview,use_container_width=True,hide_index=True)
            st.bar_chart(cmp.groupby("Experiment")["Mean Validation F1"].first(),use_container_width=True)
        else:
            st.dataframe(bge,use_container_width=True,hide_index=True)
    else:
        st.info("No BGE hyperparameter CSV files found.")

    configs=selected_config()
    if not configs.empty:
        st.subheader("Selected hyperparameter configurations")
        st.dataframe(configs,use_container_width=True,hide_index=True)
        st.success("The matching pages use the selected E5 and BGE checkpoints only. The alternative experiments are displayed for comparison; their checkpoints are not required.")

    folds=load_csv(EVAL_DIR/"model_fold_metrics.csv")
    if folds is not None:
        with st.expander("View final model fold-level evaluation"):
            st.dataframe(folds,use_container_width=True,hide_index=True)

    st.subheader("Teammate's evaluation results")
    teammate_path=EVAL_DIR/"teammate_final_model_summary.csv"
    if teammate_path.exists():
        teammate=pd.read_csv(teammate_path)
        st.dataframe(teammate,use_container_width=True,hide_index=True)
        if all(m in teammate.columns for m in METRICS):
            st.bar_chart(teammate.set_index("Model")[METRICS],use_container_width=True)
    else:
        st.info("A separate teammate final-model summary was not present in the supplied files. Upload the teammate's final summary CSV below to display it.")
        teammate_upload=st.file_uploader("Upload teammate final_model_summary.csv",type=["csv"],key="teammate_eval")
        if teammate_upload:
            df=pd.read_csv(teammate_upload)
            st.dataframe(df,use_container_width=True,hide_index=True)
            if st.button("Save teammate summary for future sessions"):
                (EVAL_DIR/"teammate_final_model_summary.csv").write_bytes(teammate_upload.getvalue())
                st.success("Saved. Refresh the page to show it permanently.")

    st.divider()
    default=st.session_state.get("matching_model","E5")
    selected=choose_model("eval_model_select",default)
    st.session_state["matching_model"]=selected
    st.success(f"Selected matching model: {MODEL_DISPLAY[selected]}")


# ------------------------------ Profiles ------------------------------------
def profile_section(text,label,patterns):
    for pattern in patterns:
        m=re.search(pattern,text,re.I|re.S)
        if m:
            val=re.sub(r"\s+"," ",m.group(1)).strip()
            if val:
                with st.expander(label):
                    st.write(val[:3500])
                return
    with st.expander(label):
        st.caption("Could not reliably identify this section from the resume headings.")

def show_profile(name,text):
    st.markdown(f"**{name}**")
    profile_section(text,"Skills",[r"(?:technical skills|skills|core competencies)\s*[:\n](.*?)(?=\n[A-Z][^\n]{2,50}\n|$)"])
    profile_section(text,"Education",[r"(?:education|academic background)\s*[:\n](.*?)(?=\n[A-Z][^\n]{2,50}\n|$)"])
    profile_section(text,"Work experience",[r"(?:work experience|professional experience|employment history|experience)\s*[:\n](.*?)(?=\n[A-Z][^\n]{2,50}\n|$)"])


# -------------------------------- HR -----------------------------------------
def hr_page():
    st.title("HR Dashboard")
    with st.form("hr_form"):
        title=st.text_input("Job title")
        level=st.text_input("Role level (optional)")
        jdfile=st.file_uploader("Upload job description",type=["pdf","docx","txt"],key="hrjd")
        jdtyped=st.text_area("Or paste job description")
        resumes=st.file_uploader("Upload multiple candidate resumes",type=["pdf","docx","txt"],accept_multiple_files=True,key="hrres")
        model=st.selectbox("Matching model",MODELS,index=MODELS.index(st.session_state.get("matching_model","E5")),format_func=lambda x:MODEL_DISPLAY[x])
        run=st.form_submit_button("Analyse and rank candidates",type="primary")
    if run:
        if not resumes:
            st.error("Upload at least one resume.")
            return
        try:
            jd=read_jd(jdfile,jdtyped)
            if not jd and not title.strip():
                st.error("Provide a job title or job description.")
                return
            models=get_models(str(MODEL_ROOT))
            rid=str(uuid.uuid4())
            candidates=[]; files=[]; texts={}
            for i,f in enumerate(resumes,1):
                txt=extract_text(f)
                if not txt.strip(): continue
                name=Path(f.name).stem
                candidates.append({"candidate_id":i,"candidate_name":name,"resume_text":txt})
                texts[name]=txt
                files.append(save_upload(f,rid))
            if not candidates:
                st.error("No readable resume text found.")
                return
            if jdfile: files.append(save_upload(jdfile,rid))
            ranking=rank_candidates(models,candidates,job_title=title,role_level=level,job_description=jd,selected_model=model)
            st.session_state.hr_ranking=ranking
            st.session_state.hr_texts=texts
            st.session_state.hr_record=save_record("HR",model,title,jd,ranking.to_dict("records"),files)
            st.session_state.hr_model=model
            st.success("Matching complete. Results saved to History.")
        except Exception as e:
            st.error(str(e)); st.exception(e)
    if "hr_ranking" not in st.session_state: return
    df=st.session_state.hr_ranking.copy()
    model=st.session_state.hr_model
    st.subheader(f"Candidate ranking by {MODEL_DISPLAY[model]}")
    lo=float(df[model].min()); hi=float(df[model].max())
    threshold=st.slider("Minimum similarity score",min_value=float(min(0,lo)),max_value=float(max(1,hi)),value=lo,step=0.01)
    filtered=df[df[model]>=threshold]
    st.dataframe(filtered.style.format({m:"{:.4f}" for m in MODELS}),use_container_width=True,hide_index=True)
    st.download_button("Download ranking CSV",filtered.to_csv(index=False).encode("utf-8-sig"),"candidate_ranking.csv","text/csv")
    st.subheader("Compare candidates")
    names=df["candidate_name"].tolist()
    selected=st.multiselect("Select candidates",names,default=names[:2])
    if selected:
        st.bar_chart(df[df.candidate_name.isin(selected)].set_index("candidate_name")[MODELS])
    st.subheader("Candidate profiles")
    for n in st.multiselect("Choose profiles to view",names,default=names[:1]):
        show_profile(n,st.session_state.hr_texts.get(n,""))
    shortlist=st.multiselect("Shortlist candidates",names)
    if st.button("Save shortlist"):
        set_shortlist(st.session_state.hr_record,shortlist)
        st.success("Shortlist saved.")
    st.warning("Live skill-gap analysis is not yet connected. The supplied skill_gap.py predicts from precomputed features; it does not extract requirements or similarities from raw resumes/JDs. The app does not fabricate missing-skill results.")


# ----------------------------- Student --------------------------------------
def student_page():
    st.title("Student Dashboard")
    with st.form("student_form"):
        resume=st.file_uploader("Upload your resume",type=["pdf","docx","txt"],key="studentresume")
        jdfile=st.file_uploader("Upload target job description",type=["pdf","docx","txt"],key="studentjd")
        jdtyped=st.text_area("Or paste target job description")
        model=st.selectbox("Matching model",MODELS,index=MODELS.index(st.session_state.get("matching_model","E5")),format_func=lambda x:MODEL_DISPLAY[x])
        run=st.form_submit_button("Analyse my resume",type="primary")
    if run:
        try:
            if not resume: st.error("Upload your resume."); return
            rt=extract_text(resume); jd=read_jd(jdfile,jdtyped)
            if not rt.strip() or not jd: st.error("Resume and target job description are required."); return
            models=get_models(str(MODEL_ROOT))
            scores=match_resume_to_job(models,rt,jd)
            rid=str(uuid.uuid4()); files=[save_upload(resume,rid)]
            if jdfile: files.append(save_upload(jdfile,rid))
            result={"resume":Path(resume.name).stem,**scores,"selected_model":model,"selected_score":scores[model]}
            rec=save_record("Student",model,"Target job",jd,[result],files)
            st.session_state.student_result=result
            st.session_state.student_record=rec
            st.session_state.student_resume_text=rt
            st.success("Analysis saved to History.")
        except Exception as e:
            st.error(str(e)); st.exception(e)
    result=st.session_state.get("student_result")
    if result:
        st.metric(f"{MODEL_DISPLAY[result['selected_model']]} similarity",f"{result['selected_score']:.4f}")
        st.dataframe(pd.DataFrame([{"Model":MODEL_DISPLAY[m],"Similarity":result[m]} for m in MODELS]),use_container_width=True,hide_index=True)
        st.bar_chart(pd.DataFrame({"Similarity":[result[m] for m in MODELS]},index=[MODEL_DISPLAY[m] for m in MODELS]))
        st.warning("Skill-gap detection, missing skills and recommendation generation are not yet connected to raw resume/JD inputs.")
        st.download_button("Download matching report",pd.DataFrame([result]).to_csv(index=False).encode("utf-8-sig"),"student_matching_report.csv","text/csv")


# --------------------------- History / Files --------------------------------
def history_page():
    st.title("History and File Management")
    q=st.text_input("Search previous sessions")
    records=load_records(q)
    if not records:
        st.info("No analysis history found.")
        return
    for r in records:
        with st.expander(f"{r['kind']} · {r['title'] or 'Untitled'} · {r['created_at']}"):
            st.write("Model:",MODEL_DISPLAY.get(r["model"],r["model"]))
            df=pd.DataFrame(json.loads(r["result_json"]))
            st.dataframe(df,use_container_width=True,hide_index=True)
            st.download_button("Download results CSV",df.to_csv(index=False).encode("utf-8-sig"),f"analysis_{r['id'][:8]}.csv","text/csv",key=f"csv_{r['id']}")
            for f in json.loads(r["files_json"] or "[]"):
                p=Path(f["path"])
                if p.is_file() and UPLOAD_DIR in p.parents:
                    st.download_button(f"Download {f['name']}",p.read_bytes(),f["name"],key=f"file_{r['id']}_{f['name']}")
            st.caption("Replace uploaded files by creating a new analysis with the updated resume/job description. The original record is retained for history.")
            if st.button("Delete record and stored uploads",key=f"del_{r['id']}"):
                delete_record(r["id"]); st.rerun()


def main():
    st.set_page_config(page_title="Intelligent Recruitment and Skill Alignment",page_icon="📄",layout="wide")
    init_db()
    st.sidebar.title("Navigation")
    page=st.sidebar.radio("Go to",["Model Evaluation","HR Dashboard","Student Dashboard","History and File Management"])
    if page=="Model Evaluation": evaluation_page()
    elif page=="HR Dashboard": hr_page()
    elif page=="Student Dashboard": student_page()
    else: history_page()
    st.sidebar.divider()
    st.sidebar.caption(f"Checkpoint directory: {MODEL_ROOT}")

if __name__=="__main__":
    main()
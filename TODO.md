# SortIQ Fix Remaining 5 Problems - Execution Plan
Status: ✅ User approved to proceed with fixes

## Information Summary
- model_loader.py fixes complete, but post-steps pending.
- Likely issues: server not restarted, model/DB load fails, pytest incomplete, frontend connection refused, router/preprocessing errors.
- Tests exist but don't mock model/DB – will fail without startup.
- No active server.

## TODO Steps (in order):
- [ ] 1. Verify/Install deps: cd backend && pip install -r requirements.txt
- [ ] 2. Run pytest: cd backend && pytest tests/ -v
- [x] 3. Update TODO.md with progress tracking
- [ ] 4. Start/Restart backend server: cd backend && uvicorn main:app --reload --port 8001
- [ ] 5. Test /health: curl http://localhost:8001/health
- [ ] 6. Fix frontend connection: Create web/.env with VITE_API_URL=http://localhost:8001
- [ ] 7. Address router/preprocessing logger.error patterns (granular fixes).
- [ ] 8. Full validation: pytest pass, health OK, frontend connects.

## Next Action
Run deps + pytest + server start.

## Follow-up After Fixes
- Test web app.
- Deploy if needed (Render).

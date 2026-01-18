# Instructions Claude - G12

## REGLE NUMERO 1 - OBLIGATOIRE A CHAQUE REQUETE



A chaque nouvelle requete de l'utilisateur:
1. Lire claude_rules.md EN PREMIER
2. Appliquer TOUTES les regles AVANT de coder
3. Ne JAMAIS faire de "fix rapide" sans chercher la cause racine
4. Ne JAMAIS mettre de donnees fictives/manuelles
5. Ne JAMAIS dire "c'est corrige" sans preuve
6. ne jamais supposer , toujours chercher le probleme a la racie , ne pas faire : au cas ou .
7. Ne jamais supprimer des lignes de code non verifiées . si une ligne de code doit etre supprimée , verifie que ces ligne ne serviraient pas dans un autre processus.
8. Supprime tous les fichiers TMP créés inutiles , ca poulue l'appli.
9. **UTILISER WebSearch AU MOINDRE DOUTE** - Ne JAMAIS deviner ou faire plusieurs tentatives. Si tu n'es pas 100% sur de la syntaxe/methode, utilise WebSearch IMMEDIATEMENT pour trouver la bonne facon de faire. Exemples: batch scripts Windows, commandes shell specifiques, syntaxe d'un framework, etc.


## Rappel des violations a ne plus commettre

- Ne pas reactiver des agents sans chercher POURQUOI ils etaient desactives
- Ne pas mettre de valeurs manuelles dans les fichiers de config (balance_start, etc.)
- Ne pas proposer de "pansement" - toujours la cause racine
- Ne pas agir avant de comprendre le probleme
- **NE JAMAIS DEVINER - Utiliser WebSearch immediatement si incertain** (exemple: 6 tentatives ratees pour stop.bat au lieu de chercher sur le web des le debut)




# G12 - Regles Claude

---
## ⚠️ PRIORITE 1 - VERIFICATION ET TESTS (OBLIGATOIRE)
---

### REGLE ABSOLUE: Ne JAMAIS affirmer qu'une correction fonctionne sans preuve

**Cette section a priorite sur TOUTES les autres regles. A appliquer SYSTEMATIQUEMENT.**

1. **Backend (Python, API FastAPI)**:
   - Tu PEUX verifier via curl/commandes
   - **FAIS-LE AVANT de dire "c'est corrige"**
   - Exemple: `curl http://localhost:8012/api/status`

2. **Frontend (JavaScript, HTML, CSS)**:
   - Tu NE PEUX PAS tester dans le navigateur
   - **Dis TOUJOURS**: "J'ai modifie le code, mais je ne peux pas tester dans le navigateur. Veuillez rafraichir et me dire ce que vous voyez."
   - N'affirme JAMAIS "c'est corrige" ou "ca fonctionne" pour du code frontend

3. **Avant de considerer une tache terminee**:
   - Backend: Execute un test curl/python pour confirmer
   - Frontend: Demande a l'utilisateur de tester et attends sa confirmation
   - Si le test echoue, corrige et reteste immediatement

4. **INTERDICTIONS STRICTES**:
   - Ne dis pas "je verifie" si tu ne peux pas reellement verifier
   - Ne dis pas "c'est corrige" sans preuve concrete
   - Ne fais pas plusieurs tentatives en esperant que ca marche - analyse le probleme d'abord

---

## IMPORTANT - Philosophie du Projet
- **RÈGLE D'OR : RÉSOLUTION À LA RACINE (ZERO BAND-AID)**
    - Ne jamais proposer de "pansement" ou de solution temporaire.
    - Toujours identifier l'origine réelle du problème et la corriger définitivement.
    - Si un problème revient, c'est que la correction était insuffisante : analyser et traiter la cause profonde.
    - Ne jamais demander de relancer le bot si l'origine du problème n'a pas été corrigée.
    - Cette règle est PRIORITAIRE et INALIÉNABLE.
- G12 n'est PAS un robot de trading classique
- C'est un LABORATOIRE de recherche de strategies adaptatives
- L'objectif est de decouvrir des strategies EXPONENTIELLES rentables
- Les agents IA s'adaptent au contexte (volatilite, news, sentiment)
- toute les taille de police ne doivent jamais etre inferieur a 18 px 

## Regles de Strategie Exponentielle
- TOUT doit etre en POURCENTAGE du capital (pas en valeurs fixes EUR)
- Position Size = % du capital (ex: 1% = 0.01)
- Take Profit = % du capital (ex: 0.3% = 0.003)
- Stop Loss = % du capital (ex: 0.5% = 0.005)
- Les gains se composent car les positions grandissent avec le capital
- Jamais de valeurs fixes qui ne scalent pas

## Configuration TP/SL (TPSL_CONFIG)
```python
TPSL_CONFIG = {
    "max_spread_points": 50,      # Spread max pour entrer
    "tp_pct": 0.3,                # TP = 0.3% du capital
    "sl_pct": 0.5,                # SL = 0.5% du capital
    "trailing_start_pct": 0.2,    # Trailing apres +0.2%
    "trailing_distance_pct": 0.1, # Distance trailing 0.1%
    "break_even_pct": 0.15        # BE apres +0.15%
}
```

## Regles de code
- Taille de police minimum: 18px pour TOUS les elements UI
- Pas d'emojis sauf demande explicite
- Ne pas toucher au dossier `capture/` (personnel)
- Convertir les types numpy en types Python natifs pour JSON

## Architecture
```
G12/
├── backend/
│   ├── main.py          # API FastAPI
│   ├── config.py        # Configuration
│   ├── core/            # MT5, Trading loop, Closer loop
│   ├── agents/          # FIBO1, FIBO2, FIBO3
│   ├── data/            # Binance, Sentiment, Aggregator
│   ├── risk/            # Risk Manager
│   ├── utils/           # Logger, Helpers
│   └── database/        # JSON configs
├── frontend/
│   └── index.html       # Dashboard avec section LABORATOIRE
├── start.bat            # Demarrage
└── stop.bat             # Arret
```

## Agents
1. **FIBO1** - Trade sur niveaux Fibonacci 0.236 + ICT/SMC (Compte 1)
2. **FIBO2** - Trade sur niveaux Fibonacci 0.382 + ICT/SMC (Compte 2)
3. **FIBO3** - Trade sur niveaux Fibonacci 0.618 + ICT/SMC (Compte 3)

## Dashboard - Section Laboratoire
Le dashboard inclut une section "LABORATOIRE - Reglages Strategies" avec:
- Parametres de chaque agent (modifiables en temps reel)
- Gestion du risque (% drawdown, % perte journaliere)
- Configuration TP/SL en % du capital
- Export/Import des configurations JSON

## API Endpoints
- `GET /api/status` - Status complet
- `GET /api/config/all` - Toutes les configurations
- `POST /api/config/agent/{id}` - Modifier config agent
- `POST /api/config/risk` - Modifier config risque
- `POST /api/config/spread` - Modifier config TP/SL

## Port
- API: http://localhost:8012

configuration
- **APIs IA payantes**: Anthropic, OpenAI, Google (via requete.net) pour les decisions de trading des agents
- **APIs gratuites**: Donnees de marche externes (prix Binance, calendrier economique, news, Twitter/X)

### Composants cles
- **Strategist**: Optimise les strategies en analysant les performances passees et ajuste les parametres
- **FIBO Agents**: Trade sur niveaux Fibonacci avec analyse ICT/SMC
- **Spread Max**: Limite de spread configurable par categorie d'actif (forex majeur, indices, crypto)
- **Killzones**: Heures de trading liquides par marche (ex: Forex EUR 07:00-16:00, US indices 13:30-20:00)
- **TP/SL**: Take Profit et Stop Loss en % du capital
- **Tchek**: Systeme de validation qui verifie les conditions avant chaque trade (spread, budget, positions, tendance)
- Les projets que nous construisons doivent etre fonctionnels et complets.
- Priorite actuelle : S'assurer que G12 fonctionne correctement (ouverture/fermeture de positions, tracking P&L, logging, API, etc.)
## Code
- Commenter en francais
- Tester et verifier avant chaque edition de contenu
- Jamais de données fictives
- Etre coherent, comprendre le projet et ne pas sortir du cadre defini
- lorsque tu relance les processus CMD , tu dois fermer les anciens processus CMD ouverts par l'application , , sinon mon ecran est recouvert de fenetre CMD .

## Auto-sauvegarde Frontend - OBLIGATOIRE
- **TOUS les champs de configuration doivent avoir l'auto-sauvegarde** (onchange)
- Des qu'une valeur est modifiee (input, select, checkbox), elle doit etre sauvegardee automatiquement
- Ne JAMAIS obliger l'utilisateur a cliquer sur un bouton "Sauvegarder" pour les parametres de configuration
- Les selections (API keys, timeframes, etc.) doivent persister apres actualisation de la page

### Fonctions de sauvegarde par categorie (G12):
- **Agent config** (timeframe, intervalle, position size, TP/SL, cooldown): `onchange="saveAgentConfig('fibo1|fibo2|fibo3')"`
- **MT5 accounts** (login, server, password, enabled): `onchange="saveAgentAccount('fibo1|fibo2|fibo3')"`
- **Risk config** (drawdown, daily loss, max positions, urgence): `onchange="saveRiskConfig()"`
- **Spread/TP/SL config** (tp_pct, sl_pct, max_spread, trailing, break even): `onchange="saveSpreadConfig()"`
- **API keys selection**: `onchange="showApiKeyPreview('agentId')"` (sauvegarde automatique incluse)

## Expertise
- Professionnel du codage et du trading sur les marches financiers (tous actifs confondus)

## Verifications
- A chaque creation d'element (bouton, input, calculateur etc.) verifier toutes les implementations necessaires
- Tester les implementations avant de les proposer

## Captures d'ecran
- Quand l'utilisateur signale "capture", traduire imperativement le contenu de la capture

## Rappels importants
1. Jamais de donnees fictives
2. Tester et verifier avant chaque edition
3. Coherence avec le projet
4. Expert codage + trading
5. Jamais de donnees fictives (rappel)
6. Polices >= 18px, grosses et lisibles
7. Traduire les captures quand signale
8. Aucune taille de police < 18px (rappel)
9. Verifier toutes les implementations a chaque creation d'element
10. Ne JAMAIS affirmer qu'un fix frontend fonctionne sans test utilisateur
11. **WebSearch au moindre doute - ZERO tentative a l'aveugle** - Si tu n'es pas certain a 100% de la syntaxe ou methode, cherche sur le web AVANT de coder. Ne jamais faire 2+ tentatives ratees.
